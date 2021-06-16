import pandas as pd
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import random
import pdb
from tqdm import tqdm

class AdaptiveTrainSet(Dataset):
    def __init__(self, belongs2, cluster_contains):
        super(AdaptiveTrainSet, self).__init__()
        self.neg_bias = 0.2
        self.belongs2 = belongs2
        self.cluster_contains = cluster_contains
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.raw_data, self.entity_to_index, self.relation_to_index = self.load_text()
        self.entity_num, self.relation_num = len(self.entity_to_index), len(self.relation_to_index)
        self.triple_num = self.raw_data.shape[0]
        # print(f'Train set: {self.entity_num} entities, {self.relation_num} relations, {self.triple_num} triplets.')
        self.pos_data = self.convert_word_to_index(self.raw_data)
        self.related_dic = self.get_related_entity()
        # self.neg_data = self.generate_neg_adaptive()

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.get_neg(self.pos_data[item])]

    def load_text(self):
        raw_data = pd.read_csv('../../data/FB15K/train.txt', sep='\t', header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False, encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: x.strip())
        head_count = Counter(raw_data['head'])
        tail_count = Counter(raw_data['tail'])
        relation_count = Counter(raw_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(word, idx) for idx, word in enumerate(entity_list)])
        relation_dic = dict([(word, idx) for idx, word in enumerate(relation_list)])
        return raw_data.values, entity_dic, relation_dic

    def convert_word_to_index(self, data):
        index_list = np.array([
            [self.entity_to_index[triple[0]], self.relation_to_index[triple[1]], self.entity_to_index[triple[2]]] for
            triple in data])
        return index_list

    def select_from_pool(self, head, rel, tail, pool, head_pool, tail_pool):
        if pool == 'all-head':
            while True:
                neg = np.random.choice(list(range(self.entity_num)), 1)
                if neg[0] not in self.related_dic[tail]:
                    break
            return neg[0]
        elif pool == 'all-tail':
            while True:
                neg = np.random.choice(list(range(self.entity_num)), 1)
                if neg[0] not in self.related_dic[head]:
                    break
            return neg[0]
        elif pool == 'head':
            try:
                for i in range(3):
                    neg = np.random.choice(head_pool, 1)
                    if neg[0] not in self.related_dic[tail]:
                        break
                    if i == 2:
                        raise ValueError
                return neg[0]
            except:
                while True:
                    neg = np.random.choice(list(range(self.entity_num)), 1)
                    if neg[0] not in self.related_dic[tail]:
                        break
                return neg[0]
        elif pool == 'tail':
            try:
                for i in range(3):
                    neg = np.random.choice(tail_pool, 1)
                    if neg[0] not in self.related_dic[head]:
                        break
                    if i == 2:
                        raise ValueError
                return neg[0]
            except:
                while True:
                    neg = np.random.choice(list(range(self.entity_num)), 1)
                    if neg[0] not in self.related_dic[head]:
                        break
                return neg[0]
        else:
            raise ValueError

    def get_neg(self, triple):
        head = triple[0]
        rel = triple[1]
        tail = triple[2]
        head_pool = self.cluster_contains[self.belongs2[head]]
        tail_pool = self.cluster_contains[self.belongs2[tail]]
        h_or_t = np.random.rand()
        if h_or_t > 0.5:
            # relpace head
            p_or_r = np.random.rand()
            if p_or_r <= self.neg_bias:
                # select from head pool
                neg_head = self.select_from_pool(head=head, rel=rel, tail=tail, pool='head', head_pool=head_pool, tail_pool=tail_pool)
                return np.array([neg_head, rel, tail])
            else:
                # select from all pool
                neg_head = self.select_from_pool(head=head, rel=rel, tail=tail, pool='all-head', head_pool=head_pool, tail_pool=tail_pool)
                return np.array([neg_head, rel, tail])
        else:
            # relpace tail
            p_or_r = np.random.rand()
            if p_or_r <= self.neg_bias:
                # select from tail pool
                neg_tail = self.select_from_pool(head=head, rel=rel, tail=tail,  pool='tail', head_pool=head_pool, tail_pool=tail_pool)
                return np.array([head, rel, neg_tail])
            else:
                # select from all pool
                neg_tail = self.select_from_pool(head=head, rel=rel, tail=tail,  pool='all-tail', head_pool=head_pool, tail_pool=tail_pool)
                return np.array([head, rel, neg_tail])

    def generate_neg_adaptive(self):
        neg_samples = []
        for triple in tqdm(self.pos_data):
            head = triple[0]
            rel = triple[1]
            tail = triple[2]
            head_pool = self.cluster_contains[self.belongs2[head]]
            tail_pool = self.cluster_contains[self.belongs2[tail]]
            h_or_t = np.random.rand()
            if h_or_t > 0.5:
                # relpace head
                p_or_r = np.random.rand()
                if p_or_r <= self.neg_bias:
                    # select from head pool
                    neg_head = self.select_from_pool(head=head, rel=rel, tail=tail, pool='head', head_pool=head_pool, tail_pool=tail_pool)
                    neg_samples.append([neg_head, rel, tail])
                else:
                    # select from all pool
                    neg_head = self.select_from_pool(head=head, rel=rel, tail=tail, pool='all-head', head_pool=head_pool, tail_pool=tail_pool)
                    neg_samples.append([neg_head, rel, tail])
            else:
                # relpace tail
                p_or_r = np.random.rand()
                if p_or_r <= self.neg_bias:
                    # select from tail pool
                    neg_tail = self.select_from_pool(head=head, rel=rel, tail=tail,  pool='tail', head_pool=head_pool, tail_pool=tail_pool)
                    neg_samples.append([head, rel, neg_tail])
                else:
                    # select from all pool
                    neg_tail = self.select_from_pool(head=head, rel=rel, tail=tail,  pool='all-tail', head_pool=head_pool, tail_pool=tail_pool)
                    neg_samples.append([head, rel, neg_tail])
        return np.array(neg_samples)

    def get_related_entity(self):
        """
        get related entities
        :return: {entity_id: {related_entity_id_1, related_entity_id_2...}}
        """
        related_dic = dict()
        for triple in self.pos_data:
            if related_dic.get(triple[0]) is None:
                related_dic[triple[0]] = {triple[2]}
            else:
                related_dic[triple[0]].add(triple[2])
            if related_dic.get(triple[2]) is None:
                related_dic[triple[2]] = {triple[0]}
            else:
                related_dic[triple[2]].add(triple[0])
        return related_dic


class TrainSet(Dataset):
    def __init__(self):
        super(TrainSet, self).__init__()
        # self.raw_data, self.entity_dic, self.relation_dic = self.load_texd()
        self.raw_data, self.entity_to_index, self.relation_to_index = self.load_text()
        self.entity_num, self.relation_num = len(self.entity_to_index), len(self.relation_to_index)
        self.triple_num = self.raw_data.shape[0]
        print(f'Train set: {self.entity_num} entities, {self.relation_num} relations, {self.triple_num} triplets.')
        self.pos_data = self.convert_word_to_index(self.raw_data)
        self.related_dic = self.get_related_entity()
        # print(self.related_dic[0], self.related_dic[479])
        self.neg_data = self.generate_neg()

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_text(self):
        raw_data = pd.read_csv('../../data/FB15K/train.txt', sep='\t', header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False, encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: x.strip())
        head_count = Counter(raw_data['head'])
        tail_count = Counter(raw_data['tail'])
        relation_count = Counter(raw_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(word, idx) for idx, word in enumerate(entity_list)])
        relation_dic = dict([(word, idx) for idx, word in enumerate(relation_list)])
        return raw_data.values, entity_dic, relation_dic

    def convert_word_to_index(self, data):
        index_list = np.array([
            [self.entity_to_index[triple[0]], self.relation_to_index[triple[1]], self.entity_to_index[triple[2]]] for
            triple in data])
        return index_list

    def generate_neg(self):
        """
        generate negative sampling
        :return: same shape as positive sampling
        """
        neg_candidates, i = [], 0
        neg_data = []
        population = list(range(self.entity_num))
        for idx, triple in enumerate(self.pos_data):
            while True:
                if i == len(neg_candidates):
                    i = 0
                    neg_candidates = random.choices(population=population, k=int(1e4))
                neg, i = neg_candidates[i], i + 1
                if random.randint(0, 1) == 0:
                    # replace head
                    if neg not in self.related_dic[triple[2]]:
                        neg_data.append([neg, triple[1], triple[2]])
                        break
                else:
                    # replace tail
                    if neg not in self.related_dic[triple[0]]:
                        neg_data.append([triple[0], triple[1], neg])
                        break

        return np.array(neg_data)

    def get_related_entity(self):
        """
        get related entities
        :return: {entity_id: {related_entity_id_1, related_entity_id_2...}}
        """
        related_dic = dict()
        for triple in self.pos_data:
            if related_dic.get(triple[0]) is None:
                related_dic[triple[0]] = {triple[2]}
            else:
                related_dic[triple[0]].add(triple[2])
            if related_dic.get(triple[2]) is None:
                related_dic[triple[2]] = {triple[0]}
            else:
                related_dic[triple[2]].add(triple[0])
        return related_dic


class TestSet(Dataset):
    def __init__(self):
        super(TestSet, self).__init__()
        self.raw_data = self.load_text()
        self.data = self.raw_data
        print(f"Test set: {self.raw_data.shape[0]} triplets")

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.data.shape[0]

    def load_text(self):
        raw_data = pd.read_csv('../../data/FB15K/test.txt', sep='\t', header=None,
                               names=['head', 'relation', 'tail'],
                               keep_default_na=False, encoding='utf-8')
        raw_data = raw_data.applymap(lambda x: x.strip())
        return raw_data.values

    def convert_word_to_index(self, entity_to_index, relation_to_index, data):
        index_list = np.array(
            [[entity_to_index[triple[0]], relation_to_index[triple[1]], entity_to_index[triple[2]]] for triple in data])
        self.data = index_list


if __name__ == '__main__':
    train_data_set = TrainSet()
    test_data_set = TestSet()
    test_data_set.convert_word_to_index(train_data_set.entity_to_index, train_data_set.relation_to_index,
                                                    test_data_set.raw_data)
    train_loader = DataLoader(train_data_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_set, batch_size=32, shuffle=True)
    for batch_idx, data in enumerate(train_loader):
        break
    # for batch_idx, (pos, neg) in enumerate(loader):
    #     # print(pos, neg)
    #     break
