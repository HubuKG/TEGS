import os
import json
import torch
from typing import List
from dataclasses import dataclass
from collections import deque
from dict import get_train_triplet_dict, get_entity_dict, EntityDict, TripletDict
from config import args
from config import logger


entity_dict: EntityDict = get_entity_dict()
train_triplet_dict: TripletDict = get_train_triplet_dict() if not args.is_test else None


def construct_mask(row_exs: List, col_exs: List = None) -> torch.tensor:
    positive_on_diagonal = col_exs is None
    num_row = len(row_exs)
    col_exs = row_exs if col_exs is None else col_exs
    num_col = len(col_exs)

    # exact match
    row_entity_ids = torch.LongTensor( [entity_dict.entity_to_idx( ex.object_id ) for ex in row_exs] )
    col_entity_ids = row_entity_ids if positive_on_diagonal else \
        torch.LongTensor( [entity_dict.entity_to_idx( ex.object_id ) for ex in col_exs] )
    # num_row x num_col
    triplet_mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0))
    if positive_on_diagonal:
        triplet_mask.fill_diagonal_(True)


    for i in range(num_row):
        subject_id, relation = row_exs[i].subject_id, row_exs[i].relation
        neighbor_ids = train_triplet_dict.get_neighbors(subject_id, relation)

        if len(neighbor_ids) <= 1:
            continue

        for j in range(num_col):
            if i == j and positive_on_diagonal:
                continue
            object_id = col_exs[j].object_id
            if object_id in neighbor_ids:
                triplet_mask[i][j] = False

    return triplet_mask


def construct_self_negative_mask(exs: List) -> torch.tensor:
    mask = torch.ones(len(exs))
    for idx, ex in enumerate(exs):
        subject_id, relation = ex.subject_id, ex.relation
        neighbor_ids = train_triplet_dict.get_neighbors(subject_id, relation)
        if subject_id in neighbor_ids:
            mask[idx] = 0
    return mask.bool()

@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.sr2object = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_triplet(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['subject_id'], ex['relation'])
            if key not in self.sr2object:
                self.sr2object[key] = set()
            self.sr2object[key].add( ex['object_id'] )
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.sr2object.get( (h, r), set() )


class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['subject_id'])
                valid_entity_ids.add(ex['object_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))

        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            subject_id, object_id = ex['subject_id'], ex['object_id']
            if subject_id not in self.graph:
                self.graph[subject_id] = set()
            self.graph[subject_id].add(object_id)
            if object_id not in self.graph:
                self.graph[object_id] = set()
            self.graph[object_id].add(subject_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, n_hop: int = 1, max_to_keep: int = 10) -> List[str]:
        if n_hop < 0:
            return []

        seen_eids = set()
        seen_eids.add( entity_id )
        queue = deque( [(entity_id, 0)] )  # (entity_id, hop_count) pair
        neighbor_ids = []

        while queue:
            curr_id, curr_hop = queue.popleft()


            if curr_hop > n_hop:
                break

            if curr_id != entity_id:
                neighbor_ids.append( curr_id )

            if len( neighbor_ids ) >= max_to_keep:
                break

            for neighbor_id in self.graph.get( curr_id, set() ):
                if neighbor_id not in seen_eids:
                    queue.append( (neighbor_id, curr_hop + 1) )
                    seen_eids.add( neighbor_id )

        return neighbor_ids

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


def reverse_triplet(obj):
    return {
        'subject_id': obj['object_id'],
        'subject': obj['object'],
        'relation': 'inverse {}'.format(obj['relation']),
        'object_id': obj['subject_id'],
        'object': obj['subject']
    }
