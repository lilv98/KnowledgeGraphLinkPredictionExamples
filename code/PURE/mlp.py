import torch
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm
from collections import Counter
import math

class FB15K_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, stage, num_ng):
        super().__init__()
        self.root = root
        self.stage = stage
        self.num_ng = num_ng
        self.train_data, self.test_data, self.entity_dic, self.relation_dic = self._read_data()
        self.entities = torch.arange(len(self.entity_dic))
        self.relations = torch.arange(len(self.relation_dic))
    
    def _read_data(self):
        train_data = pd.read_csv(self.root + 'train.txt', sep='\t', header=None,
                                names=['head', 'relation', 'tail'],
                                keep_default_na=False, encoding='utf-8')
        train_data = train_data.applymap(lambda x: x.strip())
        head_count = Counter(train_data['head'])
        tail_count = Counter(train_data['tail'])
        relation_count = Counter(train_data['relation'])
        entity_list = list((head_count + tail_count).keys())
        relation_list = list(relation_count.keys())
        entity_dic = dict([(word, idx) for idx, word in enumerate(entity_list)])
        relation_dic = dict([(word, idx) for idx, word in enumerate(relation_list)])
        train_data['head'] = train_data['head'].map(entity_dic)
        train_data['tail'] = train_data['tail'].map(entity_dic)
        train_data['relation'] = train_data['relation'].map(relation_dic)
        test_data = pd.read_csv(self.root + 'test.txt', sep='\t', header=None,
                                names=['head', 'relation', 'tail'],
                                keep_default_na=False, encoding='utf-8')
        test_data['head'] = test_data['head'].map(entity_dic)
        test_data['tail'] = test_data['tail'].map(entity_dic)
        test_data['relation'] = test_data['relation'].map(relation_dic)
        # print(f'Entities: {len(entity_dic)} \nRelations: {len(relation_dic)}')
        return torch.tensor(train_data.values), torch.tensor(test_data.values), entity_dic, relation_dic
    
    def train_sampling(self, pos):
        X = pos.tile(self.num_ng + 1, 1)
        fake_e = torch.randperm(len(self.entities))[:self.num_ng]
        fake_h = fake_e[:self.num_ng//2]
        fake_t = fake_e[self.num_ng//2:]
        X[1:self.num_ng//2 + 1, 0] = fake_h
        X[self.num_ng//2 + 1:, 2] = fake_t
        y = torch.zeros(self.num_ng + 1, 1)
        y[0, 0] = 1
        return X, y
    
    def test_sampling(self, pos):
        X = pos.tile(len(self.entity_dic), 1)
        X[:, 2] = self.entities
        return torch.cat([pos.unsqueeze(dim=0), X[:pos[2]], X[pos[2] + 1:]], dim=0)
    
    def __len__(self):
        if self.stage.lower() == 'train':
            return len(self.train_data)
        elif self.stage.lower() == 'test':
            return len(self.test_data)
        else:
            raise ValueError
    
    def __getitem__(self, idx):
        if self.stage.lower() == 'train':
            X, y = self.train_sampling(self.train_data[idx])
            return X, y
        elif self.stage.lower() == 'test':
            X = self.test_sampling(self.test_data[idx])
            return X
        else:
            raise ValueError

class LookupEmbedding(torch.nn.Module):
    def __init__(self, emb_dim, dataset):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_e = len(dataset.entity_dic)
        self.num_r = len(dataset.relation_dic)
        self.embedding_e = torch.nn.Embedding(self.num_e, self.emb_dim)
        self.embedding_r = torch.nn.Embedding(self.num_r, self.emb_dim)
    
    def forward(self, X):
        h_emb = self.embedding_e(X[:, 0])
        r_emb = self.embedding_r(X[:, 1])
        t_emb = self.embedding_e(X[:, 2])
        emb = torch.cat([h_emb, r_emb, t_emb], dim=1)
        return emb

class Generator(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

        def block(in_feat, out_feat):
            layers = [torch.nn.Linear(in_feat, out_feat)]
            layers.append(torch.nn.BatchNorm1d(out_feat))
            layers.append(torch.nn.ReLU())
            return layers

        self.model = torch.nn.Sequential(
            *block(self.emb_dim, self.emb_dim),
            *block(self.emb_dim, self.emb_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.fc_0 = torch.nn.Linear(emb_dim * 3, emb_dim)
        self.fc_1 = torch.nn.Linear(emb_dim, 1)
    
    def forward(self, emb):
        return torch.sigmoid(self.fc_1(torch.relu(self.fc_0(emb))))

def train_collate_fn(batch):
    X, y = zip(*batch)
    return torch.cat(X, dim=0), torch.cat(y, dim=0)

def test_collate_fn(batch):
    return torch.cat(batch, dim=0)

def evaluate(pred, Ks=[1, 5, 10]):
    ranks = (pred.argsort(descending=True, dim=1) == 0).nonzero(as_tuple=True)[1] + 1
    rranks = 1/ranks
    hits = []
    for K in Ks:
        hit = []
        for rank in ranks:
            if rank <= K:
                hit.append(1)
            else:
                hit.append(0)
        hits.append(sum(hit)/len(hit))
    mr = ranks.float().mean().item()
    mrr = rranks.float().mean().item()
    return mr, mrr, hits

if __name__ == '__main__':
    root = '../../data/FB15K/'
    bs = 512
    emb_dim = 50
    gpu_id = 0
    pos_ratio = 1e-5
    # num_ng = math.ceil(1.0 / (1 - 2*pos_ratio)**2)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.MSELoss()
    train_dataset = FB15K_Dataset(root=root, stage='train', num_ng=2)
    test_dataset = FB15K_Dataset(root=root, stage='test', num_ng=None)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=bs,
                                                   num_workers=8,
                                                   shuffle=True,
                                                   collate_fn=train_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=bs,
                                                  num_workers=8,
                                                  shuffle=False,
                                                  collate_fn=test_collate_fn)
    EMB = LookupEmbedding(emb_dim=emb_dim, dataset=train_dataset)
    EMB = EMB.to(device)
    D = Discriminator(emb_dim=emb_dim)
    D = D.to(device)
    optim_D = torch.optim.Adam(list(EMB.parameters()) + list(D.parameters()), lr=0.01)
    # G_head = Generator(emb_dim=emb_dim)
    # G_tail = Generator(emb_dim=emb_dim)
    # G_head = G_head.to(device)
    # G_tail = G_tail.to(device)
    for epoch in range(30):
        D.train()
        print(f'Epoch: {epoch + 1}')
        avg_loss = 0
        for X, y in tqdm(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            emb = EMB(X)
            pred = D(emb)
            loss = criterion(pred, y)
            optim_D.zero_grad()
            loss.backward()
            optim_D.step()
            avg_loss += loss.item()
        print(avg_loss/len(train_dataloader))
        with torch.no_grad():
            D.eval()
            MR = []
            MRR = []
            H1 = []
            H5 = []
            H10 = []
            for X in tqdm(test_dataloader):
                X = X.to(device)
                emb = EMB(X)
                pred = D(emb).view(-1, len(train_dataset.entity_dic))
                mr, mrr, hits = evaluate(pred)
                MR.append(mr)
                MRR.append(mrr)
                H1.append(hits[0])
                H5.append(hits[1])
                H10.append(hits[2])
            MR = round(sum(MR)/len(test_dataloader), 4)
            MRR = round(sum(MRR)/len(test_dataloader), 4)
            H1 = round(sum(H1)/len(test_dataloader), 4)
            H5 = round(sum(H5)/len(test_dataloader), 4)
            H10 = round(sum(H10)/len(test_dataloader), 4)
            print(f'MR:{MR}\nMRR:{MRR}\nH@1:{H1}\nH@5:{H5}\nH@10:{H10}')
            
        
        # z_head = torch.normal(mean=0, std=1, size=(bs, emb_dim))
        # z_tail = torch.normal(mean=0, std=1, size=(bs, emb_dim))
        # z_head = z_head.to(device)
        # z_tail = z_tail.to(device)
        # fake_head = G_head(z_head)
        # fake_tail = G_tail(z_tail)

