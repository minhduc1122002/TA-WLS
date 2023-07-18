import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from tqdm.notebook import tqdm

from utils.data_utils import define_actions
from utils.parser import config
from utils.h36motion3d import *
from model.model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_name = f'{config.model_name}_{config.missing_mode}_seed{config.seed}.pth'

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(config.seed)

def mpje_error(batch_pred, batch_gt, mask):
    batch_pred = batch_pred.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)
    mask = mask.contiguous().view(-1, 3)
    ones = torch.ones(mask.size()).to(device)
    return torch.mean(torch.norm( (ones - mask) * (batch_gt - batch_pred), 2, 1))

train_set = H36MDataSet(data_dir=config.data_path, total_seq=config.total_frame, sample_rate=2, actions=None, split="train",
                        num_missing=config.missing_joint, missing_length=config.missing_frame, missing_mode=config.missing_mode)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = H36MDataSet(data_dir=config.data_path, total_seq=config.total_frame, sample_rate=2, actions=None, split="valid",
                        num_missing=config.missing_joint, missing_length=config.missing_frame, missing_mode=config.missing_mode)
valid_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

if config.model_name == 'tawls':
  model = TAWLS(input_dim=3,
                num_frames=config.total_frame,
                num_joints=config.total_joint,
                num_heads=config.ta_n_heads,
                n_hidden=config.wls_n_hidden,
                scale_hidden=config.wls_hidden_scale,
                dropout=config.dropout)
elif config.model_name == 'tagcn':
  model = TAGCN(input_dim=3,
                num_frames=config.total_frame,
                num_joints=config.total_joint,
                num_heads=config.ta_n_heads,
                dropout=config.dropout)
elif config.model_name == 'stgcn':
  model = STGCN(input_dim=3,
                num_frames=config.total_frame,
                num_joints=config.total_joint,
                kernel_size=(3, 1),
                dropout=config.dropout)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(total_params)

optimizer = optim.Adam(model.parameters(), lr=config.base_lr, weight_decay=1e-05)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)

def train(model, optimizer, scheduler, train_loader, valid_loader):
    train_loss = []
    val_loss = []

    for epoch in range(config.n_epoch):
      running_loss = 0
      n = 0
      model.train()
      for batch in tqdm(train_loader, desc='Training', leave=False):
          input = batch[0].to(device)
          mask = batch[1].to(device)
          batch_dim = input.shape[0]
          n += batch_dim
          sequences_gt = input.view(-1, config.total_frame, config.total_joint, 3)
          mask = mask.view(-1, config.total_frame, config.total_joint, 3)
          sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
          optimizer.zero_grad()
          sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
          loss = mpje_error(sequences_predict, sequences_gt, mask)
          loss.backward()
          optimizer.step()

          running_loss += loss * batch_dim
      train_loss.append(running_loss.detach().cpu() / n)

      print('Epoch {:d} || Training Loss = {:.4f}'.format(epoch + 1, (running_loss.detach().cpu() / n)))
      model.eval()
      with torch.no_grad():
          running_loss = 0
          n = 0
          for batch in tqdm(valid_loader, desc='Validating', leave=False):
              input = batch[0].to(device)
              mask = batch[1].to(device)
              batch_dim = input.shape[0]
              n += batch_dim
              sequences_gt = input.view(-1, config.total_frame, config.total_joint, 3)
              mask = mask.view(-1, config.total_frame, config.total_joint, 3)
              sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
              optimizer.zero_grad()
              sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
              loss = mpje_error(sequences_predict, sequences_gt, mask) * (config.total_frame / config.missing_frame) * (config.total_joint / config.missing_joint)

              running_loss += loss * batch_dim
          val_loss.append(running_loss.detach().cpu() / n)
          print('Epoch {:d} || Validation MPJPE = {:.4f} \n'.format(epoch + 1, (running_loss.detach().cpu() / n)))
      scheduler.step()

      if (epoch + 1) % 5 == 0:
        print('----Saving-----')
        torch.save(model.state_dict(), os.path.join('./', config.save_name))

train(model, optimizer, scheduler, train_loader, valid_loader)