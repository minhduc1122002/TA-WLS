import torch
import torch.nn as nn
from model.layers import *

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD

class Graph():
    def __init__(self):
        self.get_edge()
        self.get_adjacency()

    def __str__(self):
        return self.A

    def get_edge(self):

          self.num_node = 22
          self_link = [(i, i) for i in range(self.num_node)]
          neighbor_link_ = [(4, 5), (5, 6), (4, 12), (8, 12),
                 (8, 9), (9, 10), (10, 11),
                 (12, 13), (13, 14), (14, 15),
                 (13, 16), (16, 17), (17, 18), (18, 19), (18, 20),
                 (13, 21), (21, 22), (22, 23), (23, 24), (23, 25)]

          neighbor_link = [(i-4,j-4) for (i,j) in neighbor_link_]
          self.edge = self_link + neighbor_link

    def get_adjacency(self):
        adjacency = np.zeros((self.num_node, self.num_node))
        for i, j in self.edge:
            adjacency[j, i] = 1
            adjacency[i, j] = 1
        normalize_adjacency = normalize_undigraph(adjacency)
        A = normalize_adjacency
        self.A = A

class TAWLS(nn.Module):
    def __init__(self,
                 input_dim,
                 num_frames,
                 num_joints,
                 num_heads,
                 n_hidden,
                 scale_hidden,
                 dropout,
                 ):

        super(TAWLS, self).__init__()

        self.block = nn.ModuleList()
        self.block.append(TAWLS_Block(input_dim, 64, num_frames, num_joints, num_heads, n_hidden, scale_hidden, dropout=dropout))
        self.block.append(TAWLS_Block(64, 32, num_frames, num_joints, num_heads, n_hidden, scale_hidden, dropout=dropout))
        self.block.append(TAWLS_Block(32, 64, num_frames, num_joints, num_heads, n_hidden, scale_hidden, dropout=dropout))
        self.out = TAGCN_Block(64, 3, num_frames, num_joints, num_heads, use_act=False, dropout=dropout, fixed=False)

    def forward(self, x):
        for block in (self.block):
            x = block(x)
        x = self.out(x, None)
        x = x.permute(0, 2, 1, 3)
        return x

class TAGCN(nn.Module):
    def __init__(self,
                 input_dim,
                 num_frames,
                 num_joints,
                 num_heads,
                 dropout
                 ):

        super(TAGCN, self).__init__()

        self.block = nn.ModuleList()
        self.block.append(TAGCN_Block(input_dim, 64, num_frames, num_joints, num_heads, use_act=True, dropout=dropout))
        self.block.append(TAGCN_Block(64, 32, num_frames, num_joints, num_heads, use_act=True, dropout=dropout))
        self.block.append(TAGCN_Block(32, 64, num_frames, num_joints, num_heads, use_act=True, dropout=dropout))
        self.out = TAGCN_Block(64, 3, num_frames, num_joints, num_heads, use_act=False, dropout=dropout)

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

    def forward(self, x):
        for block in (self.block):
            x = block(x, self.A)
        x = self.out(x, self.A)
        x = x.permute(0, 2, 1, 3)
        return x

class TAAGCN(nn.Module):
    def __init__(self,
                 input_dim,
                 num_frames,
                 num_joints,
                 num_heads,
                 dropout
                 ):

        super(TAAGCN, self).__init__()

        self.block = nn.ModuleList()
        self.block.append(TAGCN_Block(input_dim, 64, num_frames, num_joints, num_heads, use_act=True, dropout=dropout, fixed=False))
        self.block.append(TAGCN_Block(64, 32, num_frames, num_joints, num_heads, use_act=True, dropout=dropout, fixed=False))
        self.block.append(TAGCN_Block(32, 64, num_frames, num_joints, num_heads, use_act=True, dropout=dropout, fixed=False))
        self.out = TAGCN_Block(64, 3, num_frames, num_joints, num_heads, use_act=False, dropout=dropout, fixed=False)

    def forward(self, x):
        for block in (self.block):
            x = block(x, None)
        x = self.out(x, None)
        x = x.permute(0, 2, 1, 3)
        return x

class STGCN(nn.Module):
    def __init__(self,
                 input_dim,
                 num_frames,
                 num_joints,
                 kernel_size,
                 dropout,
                 ):

        super(STGCN, self).__init__()

        self.block = nn.ModuleList()
        self.block.append(STGCN_Block(input_dim, 64, num_frames, num_joints, kernel_size, use_act=True, dropout=dropout))
        self.block.append(STGCN_Block(64, 32, num_frames, num_joints, kernel_size, use_act=True, dropout=dropout))
        self.block.append(STGCN_Block(32, 64, num_frames, num_joints, kernel_size, use_act=True, dropout=dropout))
        self.out = STGCN_Block(64, 3, num_frames, num_joints, kernel_size, use_act=True, dropout=dropout)

        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

    def forward(self, x):
        for block in (self.block):
            x = block(x , self.A)
        x = self.out(x, self.A)
        x = x.permute(0, 2, 1, 3)
        return x
