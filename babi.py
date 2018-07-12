
from time import time
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from basenet import BaseNet, HPSchedule
from basenet.data import ZipDataloader
from basenet.helpers import to_numpy, set_seeds

from frog.models import babi

set_seeds(345)

# --
# IO

X_train = torch.LongTensor(np.load('data/babi/inputs_train.npy')).cuda()
q_train = torch.LongTensor(np.load('data/babi/queries_train.npy')).cuda()
y_train = torch.LongTensor(np.load('data/babi/answers_train.npy')).cuda()
X_test  = torch.LongTensor(np.load('data/babi/inputs_test.npy')).cuda()
q_test  = torch.LongTensor(np.load('data/babi/queries_test.npy')).cuda()
y_test  = torch.LongTensor(np.load('data/babi/answers_test.npy')).cuda()

y_train, y_test = y_train.squeeze(), y_test.squeeze()

uclass  = np.unique(np.hstack([y_train, y_test]))
lookup  = dict(zip(uclass, range(len(uclass))))
y_train = torch.LongTensor([lookup[yy] for yy in to_numpy(y_train)])
y_test  = torch.LongTensor([lookup[yy] for yy in to_numpy(y_test)])

num_words   = int(max(X_train.max(), X_test.max())) + 1
num_classes = int(max(y_train.max(), y_test.max())) + 1

story_width = X_train.shape[1]
query_width = q_train.shape[1]

# --
# Data

class BABIDataset(Dataset):
    def __init__(self, X, q, y):
        assert X.shape[0] == q.shape[0]
        assert X.shape[0] == y.shape[0]
        
        self.X = X
        self.q = q
        self.y = y
    
    def __getitem__(self, idx):
        return (self.X[idx], self.q[idx]), self.y[idx]
    
    def __len__(self):
        return self.X.shape[0]

train_data = BABIDataset(X=X_train, q=q_train, y=y_train)
test_data = BABIDataset(X=X_test, q=q_test, y=y_test)

train_indices, search_indices = train_test_split(range(len(X_train)), train_size=0.5)
dataloaders = {
    "train"  : ZipDataloader([
        torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=32,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        ),
        torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=32,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(search_indices),
        )
    ]),
    "test"  : DataLoader(
        test_data,
        batch_size=32,
        shuffle=True
    )
}

# --
# Run

epochs = 20
unrolled = False

arch = babi.Architecture().to('cuda')
arch.init_optimizer(
    opt=torch.optim.Adam,
    params=arch.parameters(),
    betas=(0.5, 0.999),
    lr=1e-3,
    clip_grad_norm=10.0,
)

model = babi.Network(
    X_width=story_width,
    q_width=query_width,
    num_words=num_words,
    num_classes=num_classes,
    emb_dim=20,
)
model.init_search(arch=arch, unrolled=unrolled)
model.init_optimizer(
    torch.optim.Adam,
    params=model.parameters(),
    hp_scheduler={
        "lr" : lambda progress: 1e-2,
        # "lr" : HPSchedule.linear(hp_max=0.005, epochs=epochs),
    },
    clip_grad_norm=10.0,
)
model = model.to('cuda')

model.verbose = False

t = time()
for epoch in range(epochs):
    train = model.train_epoch(dataloaders, mode='train', compute_acc=True)
    test  = model.eval_epoch(dataloaders, mode='test', compute_acc=True)
    print({
        "epoch"     : int(epoch),
        "train_acc" : float(train['acc']),
        "test_acc"  : float(test['acc']),
        "elapsed"   : time() - t,
    })
    # w0 = model._arch_get_params()[0]
    # print(w0[:,:5])


