import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_frames, num_joints):
        super(GCN, self).__init__()
        
        self.A = nn.Parameter(torch.FloatTensor(num_frames, num_joints, num_joints))
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        # self.T = nn.Parameter(torch.FloatTensor(num_joints, num_frames, num_frames)) 
        # stdv = 1. / math.sqrt(self.T.size(1))
        # self.T.data.uniform_(-stdv, stdv)

        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                (1, 1)
        )
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        # x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        x = torch.einsum('nctv,tvw->nctw', (x, self.A)).contiguous()
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x 

class TA(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_heads=8, dropout=0.1):
        super(TA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.d_head = int(out_channels / num_heads) 
        self.num_heads = num_heads

        self.qkv_conv = nn.Conv2d(self.in_channels, 3 * self.out_channels, kernel_size=1, stride=1, padding=0)

        self.attn_linear = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1)
        
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

class SCTAN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_frames,
                 num_joints,
                 num_heads,
                 dropout,
                 bias=True):
        
        super(SCTAN,self).__init__()
        
        self.gcn = GCN(in_channels, out_channels, num_frames, num_joints)
        self.ta = TA(out_channels, out_channels, num_heads=num_heads, dropout=dropout)

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

class TEN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        
        super(TEN, self).__init__()

        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.block= [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                     nn.BatchNorm2d(out_channels),
                     nn.Dropout(dropout, inplace=True)] 

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output

class Model(nn.Module):
    def __init__(self,
                 input_channels,
                 time_frame,
                 ssta_dropout,
                 num_joints,
                 n_ten_layers,
                 ten_kernel_size,
                 ten_dropout,
                 bias=True):
        
        super(Model,self).__init__()
        self.input_time_frame = time_frame
        self.num_joints = num_joints
        self.n_ten_layers = n_ten_layers

        self.sstas = nn.ModuleList()
        self.tens = nn.ModuleList()
        self.relus = nn.ModuleList()
        
        
        self.sstas.append(SCTAN(in_channels=input_channels, out_channels=64, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))
        self.sstas.append(SCTAN(in_channels=64, out_channels=32, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))
        self.sstas.append(SCTAN(in_channels=32, out_channels=64, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))
        self.sstas.append(SCTAN(in_channels=64, out_channels=3, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))                                                          

        self.tens.append(TEN(self.input_time_frame, self.input_time_frame, ten_kernel_size, ten_dropout))
        self.relus.append(nn.PReLU())

        for i in range(1, self.n_ten_layers):
            self.tens.append(TEN(self.input_time_frame, self.input_time_frame, ten_kernel_size, ten_dropout))
            self.relus.append(nn.PReLU())

    def forward(self, x):
        for ssta in (self.sstas):
            x = ssta(x)
        x = x.permute(0, 2, 1, 3)
        x = self.relus[0](self.tens[0](x))
        for i in range(1, self.n_ten_layers):
            x = self.relus[i](self.tens[i](x)) + x
        return x