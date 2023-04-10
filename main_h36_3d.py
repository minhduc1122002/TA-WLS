import os
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from tqdm.notebook import tqdm

from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.parser import args
from utils.h36motion3d import *
from model.model import TAGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = f'h36m3d_{args.missing_mode}_{args.missing_frame}_{args.missing_joint}'

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def train(model, optimizer, scheduler, train_loader, valid_loader):
    train_loss = []
    val_loss = []

    for epoch in range(args.n_epochs):
      running_loss = 0
      n = 0
      model.train()
      for batch in tqdm(train_loader, desc='Training', leave=False):
          input = batch[0].to(device)
          mask = batch[1].to(device)
          batch_dim = input.shape[0]
          n += batch_dim
          sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
          mask = mask.view(-1, args.total_frame, args.total_joint, 3)
          sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
          optimizer.zero_grad() 
          sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
          loss = loss_func(sequences_predict, sequences_gt, mask, 
                          args.total_frame, args.total_joint, args.missing_frame, args.missing_joint)
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
              sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
              mask = mask.view(-1, args.total_frame, args.total_joint, 3)
              sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
              optimizer.zero_grad() 
              sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
              loss = loss_func(sequences_predict, sequences_gt, mask, 
                          args.total_frame, args.total_joint, args.missing_frame, args.missing_joint)
               
              running_loss += loss * batch_dim
          val_loss.append(running_loss.detach().cpu() / n)
          print('Epoch {:d} || Validation Loss = {:.4f} \n'.format(epoch + 1, (running_loss.detach().cpu() / n)))
      scheduler.step()

      if (epoch + 1) % 10 == 0:
        print('----Saving-----')
        torch.save(model.state_dict(), os.path.join(args.model_path, f'{model_name}.pth'))
        plt.figure(1)
        plt.plot(train_loss, 'r', label='Train loss')
        plt.plot(val_loss, 'g', label='Val loss')
        plt.legend()
        plt.show()

def test(model):
    model.load_state_dict(torch.load(os.path.join(args.model_path, f'{model_name}.pth')))
    model.eval()
    accum_loss = 0  
    n_batches = 0
    actions = define_actions('all')
    for action in actions:
      running_loss = 0
      n = 0
      test_set = H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, sample_rate=2, actions=[action], split="test",
                             num_missing=args.missing_joint, missing_length=args.missing_frame, missing_mode=args.missing_mode)
      test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
      for batch in tqdm(test_loader, desc='Testing', leave=False):
        with torch.no_grad():
            input = batch[0].to(device)
            mask = batch[1].to(device)
            batch_dim = input.shape[0]
            n += batch_dim
            sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
            mask = mask.view(-1, args.total_frame, args.total_joint, 3)
            sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
            sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
            loss = loss_func(sequences_predict, sequences_gt, mask, 
                          args.total_frame, args.total_joint, args.missing_frame, args.missing_joint)
            running_loss += loss * batch_dim
            accum_loss += loss * batch_dim
      print('Loss For {} = {:4f} \n'.format(str(action), running_loss.item() / n))
      n_batches += n
    print('Overall MPJE = {:4f} '.format(accum_loss.item() / n_batches))

seed_everything(42)

train_set = H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, sample_rate=2, actions=None, split="train",
                        num_missing=args.missing_joint, missing_length=args.missing_frame, missing_mode=args.missing_mode)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

valid_set = H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, sample_rate=2, actions=None, split="valid",
                        num_missing=args.missing_joint, missing_length=args.missing_frame, missing_mode=args.missing_mode)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

model = TAGCN(input_channels=3, time_frame=args.total_frame, 
              ssta_dropout=0, num_joints=args.total_joint, n_ten_layers=4, ten_kernel_size=[3, 3], ten_dropout=0).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

train(model, optimizer, scheduler, train_loader, valid_loader)
test(model)
