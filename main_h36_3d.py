import os
from utils import h36motion3d
from torch.utils.data import DataLoader
from model import *
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Model(input_channels=3, time_frame=args.total_frame, 
              ssta_dropout=0, num_joints=args.total_joint, n_ten_layers=4, ten_kernel_size=[3, 3], ten_dropout=0).to(device)


model_name='h36_3d_' + str(missing_frame) + 'frames_' + str(missing_joint) + '_ckpt'

def train():
    optimizer = optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=1e-05)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35, 40], gamma=0.1)
    train_loss = []
    val_loss = []
    train_dataset = h36motion3d.H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, skip_rate=1, actions=None, split=0, num_missing=args.missing_joint, missing_length=args.missing_frame)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    vald_dataset = h36motion3d.H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, skip_rate=1, actions=None, split=1, num_missing=args.missing_joint, missing_length=args.missing_frame)
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for epoch in range(n_epoch):
      running_loss = 0
      n = 0
      model.train()
      for cnt, batch in enumerate(train_loader):
          input = batch[0].to(device)
          mask = batch[1].to(device)
          batch_dim = input.shape[0]
          n += batch_dim
          sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
          mask = mask.view(-1, args.total_frame, args.total_joint, 3)
          sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
          optimizer.zero_grad() 
          sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
          loss = loss_func(sequences_predict, sequences_gt, mask)
          if cnt % 200 == 0:
            print('[%d, %5d] Training loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
          loss.backward()  
          optimizer.step()
          
          running_loss += loss * batch_dim
      train_loss.append(running_loss.detach().cpu() / n)
      print('Epoch Training Loss: %.3f' %(running_loss.detach().cpu() / n))

      model.eval()
      with torch.no_grad():
          running_loss = 0 
          n = 0
          for cnt, batch in enumerate(vald_loader):
              input = batch[0].to(device)
              mask = batch[1].to(device)
              batch_dim = input.shape[0]
              n += batch_dim
              sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
              mask = mask.view(-1, args.total_frame, args.total_joint, 3)
              sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
              optimizer.zero_grad() 
              sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
              loss = loss_func(sequences_predict, sequences_gt, mask)
              if cnt % 200 == 0:
                print('[%d, %5d]  Validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item())) 
              running_loss += loss * batch_dim
          val_loss.append(running_loss.detach().cpu() / n)
          print('Epoch Validation Loss: %.3f' %(running_loss.detach().cpu() / n))
      scheduler.step()
      if (epoch + 1) % 10 == 0:
        print('----saving model-----')
        torch.save(model.state_dict(), os.path.join('./', model_name))
        plt.figure(1)
        plt.plot(train_loss, 'r', label='Train loss')
        plt.plot(val_loss, 'g', label='Val loss')
        plt.legend()
        plt.show()

def test():
    model.load_state_dict(torch.load(os.path.join('./', model_name)))
    model.eval()
    accum_loss = 0  
    n_batches = 0
    actions = define_actions('all')
    for action in actions:
      running_loss = 0
      n = 0
      dataset_test = H36MDataSet(data_dir=args.data_dir, total_seq=args.total_frame, skip_rate=1, actions=[action], split=2, num_missing=args.missing_joint, missing_length=args.missing_frame)
      test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
      for cnt, batch in enumerate(test_loader):
        with torch.no_grad():
            input = batch[0].to(device)
            mask = batch[1].to(device)
            batch_dim = input.shape[0]
            n += batch_dim
            sequences_gt = input.view(-1, args.total_frame, args.total_joint, 3)
            mask = mask.view(-1, args.total_frame, args.total_joint, 3)
            sequences_missing = (mask * sequences_gt).permute(0, 3, 1, 2)
            sequences_predict = model(sequences_missing).permute(0, 1, 3, 2)
            loss = loss_func(sequences_predict, sequences_gt, mask)
            running_loss += loss * batch_dim
            accum_loss += loss * batch_dim
      print('Loss at test subject for action : ' + str(action) + ' is: ' + str(running_loss/n))
      n_batches += n
    print('Overall average loss in mm is: ' + str(accum_loss/n_batches))


if __name__ == '__main__':
    if args.mode == 'train':
      train()
    elif args.mode == 'test':
      test()

