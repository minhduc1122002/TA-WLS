from torch.utils.data import Dataset
import numpy as np
from h5py import File
import scipy.io as sio
from utils.data_utils import *
from matplotlib import pyplot as plt
import torch

import os 

class H36MDataSet(Dataset):
    def __init__(self, data_dir, total_seq, sample_rate, actions=None, split="train", num_missing=3, missing_length=25, missing_mode="random"):
        self.data_path = os.path.join(data_dir,'h3.6m/dataset')

        self.split = split
        self.sample_rate = sample_rate
        self.p3d = {}
        self.data_idx = []

        self.num_missing = num_missing
        self.missing_length = missing_length
        self.seq_len = total_seq
        self.split = split
        self.missing_mode = missing_mode

        if self.split == 'train':
          self.subjects = np.array([1, 6, 7, 8, 9])
        
        elif self.split == 'valid':
          self.subjects = np.array([11])
        
        else:
          self.subjects = np.array([5])

        if actions is None:
            self.actions = ["walking", "eating", "smoking", "discussion", "directions",
                    "greeting", "phoning", "posing", "purchases", "sitting",
                    "sittingdown", "takingphoto", "waiting", "walkingdog", "walkingtogether"]
        else:
            self.actions = actions

        key = 0
        for subject in self.subjects:
            for action_idx in np.arange(len(self.actions)):
                action = self.actions[action_idx]
                if self.split == "train" or self.split == "valid":
                    for subact in [1, 2]:
                        print("Subject {0} || Action {1} || Subaction {2}".format(subject, action, subact))
                        filename = '{0}/S{1}/{2}_{3}.txt'.format(self.data_path, subject, action, subact)
                        the_sequence = readCSVasFloat(filename)
                        n, d = the_sequence.shape
                        even_list = range(0, n, self.sample_rate)
                        num_frames = len(even_list)
                        the_sequence = np.array(the_sequence[even_list, :])
                        the_sequence = torch.from_numpy(the_sequence).float().cuda()
                        the_sequence[:, 0:6] = 0
                        p3d = expmap2xyz_torch(the_sequence)
                        self.p3d[key] = p3d.view(num_frames, -1).cpu().data.numpy()
                        valid_frames = np.arange(0, num_frames - self.seq_len + 1, 1)
                        tmp_data_idx_1 = [key] * len(valid_frames)
                        tmp_data_idx_2 = list(valid_frames)
                        self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                        key += 1
                else:
                    print("Subject {0} || Action {1} || Subaction {2}".format(subject, action, 1))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.data_path, subject, action, 1)
                    the_sequence1 = readCSVasFloat(filename)
                    n, d = the_sequence1.shape
                    even_list = range(0, n, self.sample_rate)

                    num_frames1 = len(even_list)
                    the_sequence1 = np.array(the_sequence1[even_list, :])
                    the_seq1 = torch.from_numpy(the_sequence1).float().cuda()
                    the_seq1[:, 0:6] = 0
                    p3d1 = expmap2xyz_torch(the_seq1)
                    self.p3d[key] = p3d1.view(num_frames1, -1).cpu().data.numpy()

                    print("Subject {0} || Action {1} || Subaction {2}".format(subject, action, 2))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(self.data_path, subject, action, 2)
                    the_sequence2 = readCSVasFloat(filename)
                    n, d = the_sequence2.shape
                    even_list = range(0, n, self.sample_rate)
                    num_frames2 = len(even_list)
                    the_sequence2 = np.array(the_sequence2[even_list, :])
                    the_seq2 = torch.from_numpy(the_sequence2).float().cuda()
                    the_seq2[:, 0:6] = 0
                    p3d2 = expmap2xyz_torch(the_seq2)
                    self.p3d[key + 1] = p3d2.view(num_frames2, -1).cpu().data.numpy()
                    fs_sel1, fs_sel2 = find_indices_256(num_frames1, num_frames2, self.seq_len,
                                                                   input_n=self.seq_len)
                    valid_frames = fs_sel1[:, 0]
                    tmp_data_idx_1 = [key] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    valid_frames = fs_sel2[:, 0]
                    tmp_data_idx_1 = [key + 1] * len(valid_frames)
                    tmp_data_idx_2 = list(valid_frames)
                    self.data_idx.extend(zip(tmp_data_idx_1, tmp_data_idx_2))
                    key += 2

        self.dimensions_to_use = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        
    def generate_continuous_corruption(self, shape, missing_seq_len, total__missing_joints):
        mask = np.ones(shape)
        joints = np.arange(shape[1] // 3)
        if self.missing_mode == "random":
          np.random.shuffle(joints)
          counter = 0
          start = np.random.randint(0, shape[0] - missing_seq_len + 1)
          
          while counter < total__missing_joints:
            missing_joint = joints[counter]
            mask[start:start + missing_seq_len, missing_joint * 3] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 1] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 2] = 0
            counter += 1
          return mask
        elif self.missing_mode == "right_leg":
          missing_joints = [1, 2, 3, 4, 5]
          start = np.random.randint(0, shape[0] - missing_seq_len + 1)

          for missing_joint in missing_joints:
            mask[start:start + missing_seq_len, missing_joint * 3] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 1] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 2] = 0
          return mask
        elif self.missing_mode == "left_leg":
          missing_joints = [6, 7, 8, 9, 10]
          start = np.random.randint(0, shape[0] - missing_seq_len + 1)

          for missing_joint in missing_joints:
            mask[start:start + missing_seq_len, missing_joint * 3] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 1] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 2] = 0
          return mask
        elif self.missing_mode == "left_hand":
          missing_joints = [15, 16, 17, 18, 19]
          start = np.random.randint(0, shape[0] - missing_seq_len + 1)

          for missing_joint in missing_joints:
            mask[start:start + missing_seq_len, missing_joint * 3] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 1] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 2] = 0
          return mask
        elif self.missing_mode == "right_hand":
          missing_joints = [20, 21, 22, 23, 24]
          start = np.random.randint(0, shape[0] - missing_seq_len + 1)

          for missing_joint in missing_joints:
            mask[start:start + missing_seq_len, missing_joint * 3] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 1] = 0
            mask[start:start + missing_seq_len, missing_joint * 3 + 2] = 0
          return mask
        else:
          return mask

    def __len__(self):
        return np.shape(self.data_idx)[0]

    def __getitem__(self, item):
        key, start_frame = self.data_idx[item]
        fs = np.arange(start_frame, start_frame + self.seq_len)
        input = self.p3d[key][fs]
        input = input[:, self.dimensions_to_use]
        mask = self.generate_continuous_corruption(input.shape, self.missing_length, self.num_missing)
        return input.astype(np.float32), mask.astype(np.float32)
