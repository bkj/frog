
from time import time
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from basenet import BaseNet
from basenet.data import ZipDataloader
from basenet.helpers import to_numpy, set_seeds

from ...frog import FROGArchitecture, FROGSearchMixin

class Architecture(FROGArchitecture):
  def __init__(self):
    super().__init__(loss_fn=F.cross_entropy)
    
    k = 5
    n = 20
    self.w0 = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1e-3, (k, n))))
    self.w1 = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1e-3, (k, n))))
    self.w2 = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1e-3, (k, n))))
    self.w3 = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1e-3, (k, n))))
    
    self._arch_params = [self.w0, self.w1, self.w2, self.w3]


class Emb2Square(nn.Module):
    def __init__(self, width, hid_sent):
        super().__init__()
        
        self.hid_sent = hid_sent
        self.width    = width
        self.linear   = nn.Linear(width, hid_sent)
        self.linear.bias.data.zero_()
        self.linear.weight.data = torch.eye(*self.linear.weight.data.shape)
    
    def forward(self, x):
        bs, _, emb_dim = x.shape
        x = x.transpose(2, 1).contiguous()
        x = x.view(bs * emb_dim, self.width)
        x = self.linear(x)
        x = x.view(bs, emb_dim, self.hid_sent)
        x = x.transpose(2, 1).contiguous()
        return x


class QueryStoryBlock(nn.Module):
    def __init__(self, num_words, emb_dim, story_width, query_width, emb_scale=0.05):
        super().__init__()
        
        self.emb_dim  = emb_dim
        self.hid_sent = emb_dim
        
        self.story_emb = nn.Embedding(num_words, emb_dim, padding_idx=0)
        self.story_square = Emb2Square(story_width, self.hid_sent)
        
        self.query_emb  = nn.Embedding(num_words, emb_dim, padding_idx=0)
        self.query_square = Emb2Square(query_width, self.hid_sent)
        
        a = 0.05
        nn.init.uniform_(self.story_emb.weight, -a, a)
        self.story_emb.weight[0].data.zero_()
        nn.init.uniform_(self.query_emb.weight, -a, a)
        self.query_emb.weight[0].data.zero_()
        
    def _embed_story(self, x):
        bs, n_sent, n_word = x.shape
        x_flat = x.view(bs * n_sent, n_word)
        story_emb = self.story_emb(x_flat)
        story_emb = story_emb.view(bs, n_sent, n_word, self.emb_dim)
        story_emb = story_emb.sum(dim=-2)
        square_story_emb = self.story_square(story_emb)
        return square_story_emb, story_emb.sum(-2)
    
    def _embed_query(self, q):
        bs, n_word_query = q.shape
        query_emb = self.query_emb(q)
        square_query_emb = self.query_square(query_emb)
        return square_query_emb, query_emb.sum(dim=-2)
    
    def forward(self, x, q):
        square_q, flat_q = self._embed_query(q)
        square_x, flat_x = self._embed_story(x)
        return flat_x, square_x, flat_q, square_q


def matvec_op(vs, ms, params):
    vl_params, ml_params, vr_params, mr_params, vs_params = params
    
    vl_weights = F.softmax(vl_params[:len(vs) + 1], dim=-1)[1:]
    ml_weights = F.softmax(ml_params[:len(ms) + 1], dim=-1)[1:]
    vl_hat     = sum(v * w for v,w in zip(vs, vl_weights))
    ml_hat     = sum(m * w for m,w in zip(ms, ml_weights))
    out_left   = torch.bmm(ml_hat, vl_hat.unsqueeze(-1)).squeeze()
    
    vr_weights = F.softmax(vr_params[:len(vs) + 1], dim=-1)[1:]
    mr_weights = F.softmax(mr_params[:len(ms) + 1], dim=-1)[1:]
    vr_hat     = sum(v * w for v,w in zip(vs, vr_weights))
    mr_hat     = sum(m * w for m,w in zip(ms, mr_weights))
    out_right  = torch.bmm(vr_hat.unsqueeze(1), mr_hat).squeeze()
    
    vs_weights   = F.softmax(vs_params[:len(vs) + 1], dim=-1)[1:]
    out_straight = sum(v * w for v,w in zip(vs, vs_weights))
    
    out = out_left + out_right + out_straight
    return out, F.softmax(out, dim=-1)


class Network(FROGSearchMixin, BaseNet):
    def __init__(self, X_width, q_width, num_words, num_classes, emb_dim=64, **kwargs):
        super().__init__(**kwargs)
        
        self.emb_dim  = emb_dim
        self.hid_sent = emb_dim
        
        self.story_block_0 = QueryStoryBlock(num_words, emb_dim, X_width, q_width)
        self.story_block_1 = QueryStoryBlock(num_words, emb_dim, X_width, q_width)
        self.story_block_2 = QueryStoryBlock(num_words, emb_dim, X_width, q_width)
        self.story_block_3 = QueryStoryBlock(num_words, emb_dim, X_width, q_width)
        
        self.classifier = nn.Linear(emb_dim, num_classes)
    
    def forward(self, data):
        x, q = data
        
        params = self._arch_get_params()
        
        # --
        # Block 0
        
        flat_x0, square_x0, flat_q0, square_q0 = self.story_block_0(x, q)
        out0, soft0 = matvec_op(
            vs=(flat_x0, flat_q0),
            ms=(square_x0, square_q0),
            params=params[0],
        )
        
        # --
        # Block 1
        
        flat_x1, square_x1, flat_q1, square_q1 = self.story_block_1(x, q)
        out1, soft1 = matvec_op(
            vs=(flat_x1, flat_q1, out0, soft0),
            ms=(square_x1, square_q1),
            params=params[1],
        )
        
        # # --
        # # Block 2
        
        # flat_x2, square_x2, flat_q2, square_q2 = self.story_block_2(x, q)
        # out2, soft2 = matvec_op(
        #     vs=(flat_x2, flat_q2, out1, soft1, out0, soft0),
        #     ms=(square_x2, square_q2),
        #     # vs=(flat_q0,),
        #     # ms=(square_x2,),
        #     params=params[2],
        # )
        
        # # --
        # # Block 3
        
        # flat_x3, square_x3, flat_q3, square_q3 = self.story_block_3(x, q)
        # out3, _ = matvec_op(
        #     vs=(flat_x3, flat_q3, out2, soft2, out1, soft1, out0, soft0),
        #     ms=(square_x3, square_q3),
        #     # vs=(soft2,),
        #     # ms=(square_x3,),
        #     params=params[3],
        # )
        
        out = out1
        return self.classifier(out)
