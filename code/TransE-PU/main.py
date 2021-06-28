import torch
import pandas as pd
import pdb
import numpy as np
from tqdm import tqdm
from collections import Counter
import math
import random
import os

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
        return X
    
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
            X = self.train_sampling(self.train_data[idx])
            return X
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
        self.embedding_e = torch.nn.Embedding.from_pretrained(
            torch.empty(self.num_e, self.emb_dim).uniform_(-6 / math.sqrt(self.emb_dim), 6 / math.sqrt(self.emb_dim)), freeze=False)
        self.embedding_r = torch.nn.Embedding.from_pretrained(
            torch.empty(self.num_r, self.emb_dim).uniform_(-6 / math.sqrt(self.emb_dim), 6 / math.sqrt(self.emb_dim)), freeze=False)
        relation_norm = torch.norm(self.embedding_r.weight.data, p=2, dim=1, keepdim=True)
        self.embedding_r.weight.data = self.embedding_r.weight.data / relation_norm
    
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
        self.emb_dim = emb_dim
    
    def forward(self, emb):
        h_emb = emb[:, :self.emb_dim]
        r_emb = emb[:, self.emb_dim:self.emb_dim*2]
        t_emb = emb[:, self.emb_dim*2:]
        distance = h_emb + r_emb - t_emb
        return torch.norm(distance, p=2, dim=1)


def my_collate_fn(batch):
    return torch.cat(batch, dim=0)

def metrics(pred, Ks=[1, 5, 10]):
    ranks = (pred.argsort(descending=False, dim=1) == 0).nonzero(as_tuple=True)[1] + 1
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

def get_ng_G(X_emb):
    # TODO: consider num_ng_G
    z_head = torch.normal(mean=0, std=0.01, size=(bs, emb_dim))
    z_tail = torch.normal(mean=0, std=0.01, size=(bs, emb_dim))
    z_head = z_head.to(device)
    z_tail = z_tail.to(device)
    fake_head = G_head(z_head)
    fake_tail = G_tail(z_tail)
    pos_emb = X_emb[torch.arange(start=0, end=len(X_emb), step=num_ng_exist+1)]
    replace_fake_head = torch.cat([fake_head, pos_emb[:, emb_dim:]], dim=1).unsqueeze(dim=1)
    replace_fake_tail = torch.cat([pos_emb[:, :emb_dim*2], fake_tail], dim=1).unsqueeze(dim=1)
    ng_G = torch.cat([replace_fake_head, replace_fake_tail], dim=1)
    X_emb = torch.cat([X_emb.view(bs, -1, emb_dim*3), ng_G], dim=1).view(-1, emb_dim*3)
    # y = torch.cat([y.view(bs, -1), torch.zeros(bs, 2).to(device)], dim=1).view(-1, 1)
    return X_emb

def D_loss(pred):
    pred = pred.view(bs, -1)
    mid = torch.sigmoid(pred[:, 0].unsqueeze(-1) - pred[:, 1:3] + margin)
    loss_P = pos_ratio * (mid.sum() - (1 - mid).sum())
    margins = pred[:, 0].unsqueeze(-1) - pred[:, 1:3] + margin
    return torch.relu(margins).sum() + loss_P

def G_loss(pred):
    pred = pred.view(bs, -1)
    margins = pred[:, 3:] - pred[:, 0].unsqueeze(-1) + margin
    return torch.relu(margins).sum()

def evaluate(epoch):
    with torch.no_grad():
        D.eval()
        G_head.eval()
        G_tail.eval()
        MR = []
        MRR = []
        H1 = []
        H5 = []
        H10 = []
        for X in tqdm(test_dataloader):
            X = X.to(device)
            pred = D(EMB(X)).view(-1, len(train_dataset.entity_dic))
            mr, mrr, hits = metrics(pred)
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

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything()
    root = '../../data/FB15K/'
    bs = 256
    emb_dim = 50
    margin = 1
    gpu_id = 0
    pos_ratio = 1e-2
    num_ng_G = math.ceil(1.0 / (1 - 2*pos_ratio)**2)
    num_ng_exist = 2
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    # criterion = torch.nn.BCELoss()
    train_dataset = FB15K_Dataset(root=root, stage='train', num_ng=num_ng_exist)
    test_dataset = FB15K_Dataset(root=root, stage='test', num_ng=None)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=bs,
                                                   num_workers=8,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=my_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                  batch_size=bs,
                                                  num_workers=8,
                                                  shuffle=False,
                                                  collate_fn=my_collate_fn)
    EMB = LookupEmbedding(emb_dim=emb_dim, dataset=train_dataset)
    EMB = EMB.to(device)
    D = Discriminator(emb_dim=emb_dim)
    D = D.to(device)
    optim_D = torch.optim.Adam(list(EMB.parameters()) + list(D.parameters()), lr=0.01)
    G_head = Generator(emb_dim=emb_dim)
    G_tail = Generator(emb_dim=emb_dim)
    G_head = G_head.to(device)
    G_tail = G_tail.to(device)
    optim_G_head = torch.optim.Adam(G_head.parameters(), lr=0.01)
    optim_G_tail = torch.optim.Adam(G_tail.parameters(), lr=0.01)

    for epoch in range(30):
        D.train()
        G_head.train()
        G_tail.train()
        print(f'\nEpoch: {epoch + 1}')
        print('Training D:')
        for _ in range(1):
            avg_loss = 0
            for X in tqdm(train_dataloader):
                X = X.to(device)
                X_emb = EMB(X)
                X_emb = get_ng_G(X_emb)
                pred = D(X_emb)
                loss = D_loss(pred)
                optim_D.zero_grad()
                loss.backward()
                optim_D.step()
                avg_loss += loss.item()
            print(round(avg_loss/len(train_dataloader), 4))
        # print('Training G:')
        # for _ in range(1):
        #     avg_loss = 0
        #     for X in tqdm(train_dataloader):
        #         X = X.to(device)
        #         X_emb = EMB(X)
        #         X_emb = get_ng_G(X_emb)
        #         # pred = D(X_emb).view(bs, -1)[:, 3:]
        #         # loss = criterion(pred, torch.ones_like(pred).to(device))
        #         pred = D(X_emb)
        #         loss = G_loss(pred)
        #         optim_G_head.zero_grad()
        #         optim_G_tail.zero_grad()
        #         loss.backward()
        #         optim_G_head.step()
        #         optim_G_tail.step()
        #         avg_loss += loss.item()
        #     print(round(avg_loss/len(train_dataloader), 4))
        print('Evaluating:')
        evaluate(epoch)
