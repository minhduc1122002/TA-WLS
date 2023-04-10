import torch
import torch.nn as nn
from model.layers import *

class TAGCN(nn.Module):
    def __init__(self,
                 input_channels,
                 time_frame,
                 ssta_dropout,
                 num_joints,
                 n_ten_layers,
                 ten_kernel_size,
                 ten_dropout,
                 bias=True):
        
        super(TAGCN,self).__init__()
        self.input_time_frame = time_frame
        self.num_joints = num_joints
        self.n_ten_layers = n_ten_layers

        self.tagcns = nn.ModuleList()
        self.tens = nn.ModuleList()
        self.relus = nn.ModuleList()
        
        
        self.tagcns.append(TAGCN_Block(in_channels=input_channels, out_channels=64, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout, use_pe=True))
        self.tagcns.append(TAGCN_Block(in_channels=64, out_channels=32, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))
        self.tagcns.append(TAGCN_Block(in_channels=32, out_channels=64, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))
        self.tagcns.append(TAGCN_Block(in_channels=64, out_channels=3, num_frames=self.input_time_frame, num_joints=self.num_joints, num_heads=1, dropout=ssta_dropout))                                                          

        self.tens.append(TEN(self.input_time_frame, self.input_time_frame, ten_kernel_size, ten_dropout))
        self.relus.append(nn.PReLU())

        for i in range(1, self.n_ten_layers):
            self.tens.append(TEN(self.input_time_frame, self.input_time_frame, ten_kernel_size, ten_dropout))
            self.relus.append(nn.PReLU())

    def forward(self, x):
        for tagcn in (self.tagcns):
            x = tagcn(x)
        x = x.permute(0, 2, 1, 3)
        x = self.relus[0](self.tens[0](x))
        for i in range(1, self.n_ten_layers):
            x = self.relus[i](self.tens[i](x)) + x
        return x
