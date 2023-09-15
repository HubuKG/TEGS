import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet import reverse_triplet
from triplet import construct_mask, construct_self_negative_mask
from dict import get_entity_dict, get_link_graph, get_tokenizer
from config import logger

entity_dict = get_entity_dict()
if args.use_link_graph:
    get_link_graph()


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(subject_id: str, object_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids( subject_id )
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != object_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)


class Example:

    def __init__(self, subject_id, relation, object_id, **kwargs):
        self.subject_id = subject_id
        self.object_id = object_id
        self.relation = relation

    @property
    def subject_desc(self):
        if not self.subject_id:
            return ''
        return entity_dict.get_entity_by_id( self.subject_id ).entity_desc

    @property
    def object_desc(self):
        return entity_dict.get_entity_by_id( self.object_id ).entity_desc

    @property
    def subject(self):
        if not self.subject_id:
            return ''
        return entity_dict.get_entity_by_id( self.subject_id ).entity

    @property
    def object(self):
        return entity_dict.get_entity_by_id( self.object_id ).entity

    def vectorize(self) -> dict:
        subject_desc, object_desc = self.subject_desc, self.object_desc
        if args.use_link_graph:
            if len(subject_desc.split()) < 20:
                subject_desc += ' ' + get_neighbor_desc( subject_id=self.subject_id, object_id=self.object_id )
            if len(object_desc.split()) < 20:
                object_desc += ' ' + get_neighbor_desc( subject_id=self.object_id, object_id=self.subject_id )

        subject_word = _parse_entity_name( self.subject )
        subject_text = _concat_name_desc(subject_word, subject_desc)
        hr_encoded_inputs = _custom_tokenize(text=subject_text,
                                             text_pair=self.relation)

        subject_encoded_inputs = _custom_tokenize(text=subject_text)

        object_word = _parse_entity_name(self.object)
        object_encoded_inputs = _custom_tokenize(text=_concat_name_desc(object_word, object_desc))

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'object_token_ids': object_encoded_inputs['input_ids'],
                'object_token_type_ids': object_encoded_inputs['token_type_ids'],
                'subject_token_ids': subject_encoded_inputs['input_ids'],
                'subject_token_type_ids': subject_encoded_inputs['token_type_ids'],
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            examples.append(Example(**obj))
        if add_backward_triplet:
            examples.append(Example(**reverse_triplet(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    object_token_ids, object_mask = to_indices_and_mask(
        [torch.LongTensor(ex['object_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    object_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['object_token_type_ids']) for ex in batch_data],
        need_mask=False)

    subject_token_ids, subject_mask = to_indices_and_mask(
        [torch.LongTensor(ex['subject_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    subject_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['subject_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'object_token_ids': object_token_ids,
        'object_mask': object_mask,
        'object_token_type_ids': object_token_type_ids,
        'subject_token_ids': subject_token_ids,
        'subject_mask': subject_mask,
        'subject_token_type_ids': subject_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
