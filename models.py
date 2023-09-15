from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet import construct_mask


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    sr_vector: torch.tensor
    object_vector: torch.tensor
    distances:torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained("./data/debert-base-uncased")
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.offset = 0
        self.sr_bert = AutoModel.from_pretrained( "./data/debert-base-uncased" )
        self.object_bert = deepcopy( self.sr_bert )

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, sr_token_ids, sr_mask, sr_token_type_ids,
                object_token_ids, object_mask, object_token_type_ids,
                subject_token_ids, subject_mask, subject_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding( object_token_ids=object_token_ids,
                                               object_mask=object_mask,
                                               object_token_type_ids=object_token_type_ids )

        sr_vector = self._encode( self.sr_bert,
                                  token_ids=sr_token_ids,
                                  mask=sr_mask,
                                  token_type_ids=sr_token_type_ids )

        object_vector = self._encode( self.object_bert,
                                    token_ids=object_token_ids,
                                    mask=object_mask,
                                    token_type_ids=object_token_type_ids )

        subject_vector = self._encode( self.object_bert,
                                    token_ids=subject_token_ids,
                                    mask=subject_mask,
                                    token_type_ids=subject_token_type_ids )

        # DataParallel only support tensor/dict
        return {'sr_vector': sr_vector,
                'object_vector': object_vector,
                'subject_vector': subject_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        sr_vector, object_vector = output_dict['sr_vector'], output_dict['object_vector']
        batch_size = sr_vector.size(0)
        labels = torch.arange(batch_size).to(sr_vector.device)
        edu_distances = torch.cdist(sr_vector, object_vector)
        n_logits = sr_vector.mm(object_vector.t())
        edu_t=0.1
        logits=n_logits - edu_t*edu_distances
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        return {'logits': logits,
                'labels': labels,
                'edu_distances': edu_distances,
                'inv_t': self.log_inv_t.detach().exp(),
                'sr_vector': sr_vector.detach(),
                'object_vector': object_vector.detach()}

    @torch.no_grad()
    def predict_ent_embedding(self, object_token_ids, object_mask, object_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode( self.object_bert,
                                    token_ids=object_token_ids,
                                    mask=object_mask,
                                    token_type_ids=object_token_type_ids )
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector