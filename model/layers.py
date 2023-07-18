import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()

        layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )

        self.layer = layer

    def forward(self, x):
        return self.layer(x)

class KernelMLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers, scale, dropout=0.0, residual=False):
        super().__init__()

        hidden_dim = int(scale * max(in_dim, out_dim))
        layers = [MLPLayer(in_dim, out_dim, dropout)]

        for _ in range(n_layers - 1):
            layers.append(MLPLayer(hidden_dim, hidden_dim, dropout))

        # layers.append(MLPLayer(hidden_dim, out_dim, dropout))

        self.layers = nn.ModuleList(layers)
        self.do_residual = residual

        if self.do_residual:
          if in_dim == out_dim:
              self.residual_layer = nn.Sequential()
          else:
              self.residual_layer = nn.Conv2d(in_dim, out_dim, 1)

    def forward(self, x):
        shape = x.shape

        _x = x
        for layer in self.layers:
            x = layer(x)

        if self.do_residual:
            x += self.residual_layer(_x)

        return x

class WLSMLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_frames, num_joints, n_hidden, scale_hidden, dropout=0.0, residual=False):
        super().__init__()
        self.out_dim = out_dim + (out_dim % 2)
        self.transform = KernelMLP(in_dim, self.out_dim // 2, n_hidden, scale_hidden, dropout, residual=residual)

        self.A = nn.Parameter(torch.FloatTensor(num_frames, num_joints, num_joints))
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_trans = self.transform(x)

        x_update = torch.einsum('nctv,tvw->nctw', (x_trans, self.A)).contiguous()
        features = torch.cat([x_update, x_trans], 1)
        return features


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_frames, num_joints, use_act, dropout):
        super(GCN, self).__init__()

        self.A = nn.Parameter(torch.FloatTensor(num_frames, num_joints, num_joints))
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1)
        )
        self.use_act = use_act
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.einsum('nctv,tvw->nctw', (x, self.A)).contiguous()
        x = self.conv(x)
        x = self.norm(x)
        if self.use_act:
          x = self.act(x)
        return x

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, num_frames, num_joints, use_act, dropout):
        super(GraphConvolution, self).__init__()

        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1)
        )

        self.use_act = use_act
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, A):
        x = torch.einsum('nctv,vw->nctw', (x, A)).contiguous()
        x = self.conv(x)
        x = self.norm(x)
        if self.use_act:
          x = self.act(x)
        return x

class TA(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_frames=25, num_joints=18, num_heads=8, dropout=0.1):
        super(TA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.d_head = int(out_channels / num_heads)
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(self.in_channels, 3 * self.in_channels, kernel_size=1, stride=1, padding=0)

        self.attn_linear = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1)

        self.concat_bn = nn.BatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop_out = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        # Input x
        # (batch_size, channels, frames, joint)
        N, C, T, V = x.size()

        # (batch_size * joint, channels, 1, frames)
        x = x.permute(0, 3, 1, 2).reshape(-1, C, 1, T)
        # (batch_size * joint, channels, 1, frame_block)
        N_block, C, _, T_block = x.shape

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.d_head * self.num_heads, self.d_head * self.num_heads, self.num_heads)
        # (batch_size * joint, head, d, frame_block)
        B, self.num_heads, C, T_block = flat_q.size()
        # (batch_size * joint, head, frame_block, frame_block)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        weights = F.softmax(logits, dim=-1)
        # (batch_size * joint, head, frame_block, d)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        # (batch_size * joint, head, 1, frame_block, d)
        attn_out = torch.reshape(attn_out, (B, self.num_heads, 1, T_block, self.d_head))
        # (batch_size * joint, head, d, 1, frames)
        attn_out = attn_out.permute(0, 1, 4, 2, 3)
        # (batch_size * joint, head * d, 1, frames)
        attn_out = self.combine_heads_2d(attn_out)
        # (batch_size, joint, head * d, frames)
        attn_out = attn_out.reshape(N, V, -1, T).permute(0, 2, 3, 1)
        attn_out = self.concat_bn(attn_out)
        attn_out = self.attn_linear(attn_out)
        attn_out = self.bn(attn_out)
        attn_out = self.drop_out(attn_out)
        return attn_out

    def compute_flat_qkv(self, x, dk, dv, num_heads):
        qkv = self.qkv_conv(x)
        N, _, T, V = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, num_heads)
        k = self.split_heads_2d(k, num_heads)
        v = self.split_heads_2d(v, num_heads)

        dkh = dk // num_heads
        q = q / (dkh ** 0.5)
        flat_q = torch.reshape(q, (N, num_heads, dk // num_heads, T * V))
        flat_k = torch.reshape(k, (N, num_heads, dk // num_heads, T * V))
        flat_v = torch.reshape(v, (N, num_heads, dv // num_heads, T * V))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, num_heads):
        N, C, T, V = x.size()
        ret_shape = (N, num_heads, C // num_heads, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        # (batch_size * joint, head, d, 1, frames)
        N, num_heads, dv, T, V = x.size()
        ret_shape = (N, num_heads * dv, T, V)
        return torch.reshape(x, ret_shape)

class TCN(nn.Module):
    def __init__(self, out_channels=64, kernel_size=(3, 1), stride=1, dropout=0.0):
        super(TCN, self).__init__()
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, x):
        x = self.tcn(x)
        return x

class TAWLS_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_frames,
                 num_joints,
                 num_heads,
                 n_hidden,
                 scale_hidden,
                 dropout):

        super(TAWLS_Block, self).__init__()

        self.gcn = WLSMLPLayer(in_channels, out_channels, num_frames, num_joints, n_hidden=n_hidden, scale_hidden=scale_hidden, dropout=dropout, residual=False)
        self.ta = TA(out_channels, out_channels, num_frames, num_joints, num_heads=num_heads, dropout=dropout)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.ta(x)
        x = x + res
        x = self.relu(x)
        return x

class TAGCN_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_frames,
                 num_joints,
                 num_heads,
                 use_act,
                 dropout,
                 fixed=True):

        super(TAGCN_Block,self).__init__()
        self.fixed = fixed
        if self.fixed:
          self.gcn = GraphConvolution(in_channels, out_channels, num_frames, num_joints, use_act=use_act, dropout=dropout)
        else:
          self.gcn = GCN(in_channels, out_channels, num_frames, num_joints, use_act=use_act, dropout=dropout)

        self.ta = TA(out_channels, out_channels, num_frames, num_joints, num_heads=num_heads, dropout=dropout)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        if self.fixed:
          x = self.gcn(x, A)
        else:
          x = self.gcn(x)
        x = self.ta(x)
        x = x + res
        x = self.relu(x)
        return x

class STGCN_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_frames,
                 num_joints,
                 kernel_size,
                 use_act,
                 dropout):

        super(STGCN_Block,self).__init__()

        self.gcn = GraphConvolution(in_channels, out_channels, num_frames, num_joints, use_act=use_act, dropout=dropout)
        self.tcn = TCN(out_channels, kernel_size=kernel_size, stride=1, dropout=dropout)

        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(1, 1)
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

        self.relu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x