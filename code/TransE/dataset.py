import pandas as pd
import numpy as np
from collections import Counter
import torch
import random
import pdb


class TrainSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.raw_data, self.entity_to_index, self.relation_to_index = self.load_data()
        self.entity_num, self.relation_num = len(self.entity_to_index), len(self.relation_to_index)
        self.triple_num = self.raw_data.shape[0]
        print(f'Train set: {self.entity_num} entities, {self.relation_num} relations, {self.triple_num} triplets.')
        self.pos_data = self.convert_word_to_index(self.raw_data)
        self.related_dic = self.get_related_entity()
        self.neg_data = self.generate_neg()

    def __len__(self):
        return self.triple_num

    def __getitem__(self, item):
        return [self.pos_data[item], self.neg_data[item]]

    def load_data(self):
        raw_data = pd.read_csv('./fb15k/freebase_mtr100_mte100-train.txt', sep='\t', header=None,
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


class DataSet(torch.utils.data.Dataset):
    def __init__(self, data_root, num_ng, split):
        super().__init__()
        self.split = split
        self.train_data, self.test_data = self.load_data(data_root)
        self.nodes = list(set(self.train_data[0].unique()) | set(self.train_data[2].unique()) | set(self.test_data[0].unique()) | set(self.test_data[2].unique()))
        self.rels = list(set(self.train_data[1].unique()) | set(self.test_data[1].unique()))
        nodes_dict = dict(zip(self.nodes, range(len(self.nodes))))
        rels_dict = dict(zip(self.rels, range(len(self.rels))))
        self.train_data[0], self.train_data[1], self.train_data[2] = self.train_data[0].map(nodes_dict), self.train_data[1].map(rels_dict), self.train_data[2].map(nodes_dict)
        self.test_data[0], self.test_data[1], self.test_data[2] = self.test_data[0].map(nodes_dict), self.test_data[1].map(rels_dict), self.test_data[2].map(nodes_dict)
        self.train_data = torch.tensor(self.train_data.values)
        self.test_data = torch.tensor(self.test_data.values)
        print(f"Train set: {self.train_data.shape[0]} triplets")
        print(f"Test set: {self.test_data.shape[0]} triplets")
        self.num_ng = num_ng
        self.related_dict = self._get_related_entity()
        self.negative_sampling()

    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_data[idx]
        elif self.split == 'test':
            return self.test_data[idx]
        else:
            raise ValueError('Split = train / test')

    def __len__(self):
        if self.split == 'train':
            return len(self.train_data)
        elif self.split == 'test':
            return len(self.test_data)
        else:
            raise ValueError('Split = train / test')

    def load_data(self, data_root):
        train_data = pd.read_csv(data_root + 'train.txt', sep='\t', header=None)
        test_data = pd.read_csv(data_root + 'test.txt', sep='\t', header=None)
        train_data = train_data.applymap(lambda x: x.strip())
        test_data = test_data.applymap(lambda x: x.strip())
        return train_data, test_data
    
    def _get_related_entity(self):
        related_dict = dict()
        for triple in self.train_data:
            if related_dict.get(triple[0].item()) is None:
                related_dict[triple[0].item()] = {triple[2].item()}
            else:
                related_dict[triple[0].item()].add(triple[2].item())
            if related_dict.get(triple[2].item()) is None:
                related_dict[triple[2].item()] = {triple[0].item()}
            else:
                related_dict[triple[2].item()].add(triple[0].item())
        return related_dict
    
    def negative_sampling(self):
        num_gen = len(self.train_data) * self.num_ng
        neg_samples = torch.tile(self.train_data, (self.num_ng, 1))
        labels = torch.zeros(len(self.train_data) * (self.num_ng + 1))
        labels[:len(self.train_data)] = 1
        values = np.random.randint(len(self.nodes), size=num_gen)
        choices = np.random.uniform(size=num_gen)
        sub = choices > 0.5
        obj = choices <= 0.5
        pdb.set_trace()
        neg_samples[sub, 0] = torch.tensor(values[sub])
        neg_samples[obj, 2] = torch.tensor(values[obj])

        return np.concatenate((pos_samples, neg_samples)), labels



if __name__ == '__main__':
    data_root = '/home/tangz0a/workspace/KG/KnowledgeGraphLinkPredictionExamples/data/FB15k-237/'
    train_data = DataSet(data_root=data_root, num_ng=1, split='train')
    print(train_data[0])

