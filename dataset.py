from scipy import interpolate
import random
import os
import os.path
import h5py
import cv2
import glob
import numpy as np
import torch
import torch.utils.data as udata
from utilities import Im2Patch

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

def process_data(patch_size, stride, path='data', mode='train'):
    if mode == 'train':
        print("\nprocess training set ...\n")
        train_num = 1
        h5f = h5py.File('train.h5', 'w')
        filenames_hyper = glob.glob(os.path.join(path,'Train_Spectral','*.mat'))
        filenames_rgb = glob.glob(os.path.join(path,'Train_RGB','*.png'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        for i in range(len(filenames_hyper)):
            print("\n")
            print([filenames_hyper[i], filenames_rgb[i]])
            # load hyperspectral image
            mat =  h5py.File(filenames_hyper[i],'r')
            hyper = np.float32(np.array(mat['rad']))
            hyper = np.transpose(hyper, [0,2,1])
            hyper = normalize(hyper, max_val=4095., min_val=0.)
            mat.close()
            # load rgb image
            rgb =  cv2.imread(filenames_rgb[i])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, [2,0,1])
            rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
            # creat patches
            patches_hyper = Im2Patch(hyper, win=patch_size, stride=stride)
            patches_rgb = Im2Patch(rgb, win=patch_size, stride=stride)
            # add data
            for i in range(patches_hyper.shape[3]):
                print("generate training sample #%d" % train_num)
                sub_hyper = patches_hyper[:,:,:,i]
                sub_rgb = patches_rgb[:,:,:,i]
                data = np.concatenate((sub_hyper,sub_rgb), 0)
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
        h5f.close()
        print("\ntraining set: # samples %d\n" % (train_num-1))
    if mode == 'test':
        print("\nprocess test set ...\n")
        test_num = 1
        h5f = h5py.File('test.h5', 'w')
        filenames_hyper = glob.glob(os.path.join(path,'Test_Spectral','*.mat'))
        filenames_rgb = glob.glob(os.path.join(path,'Test_RGB','*.png'))
        filenames_hyper.sort()
        filenames_rgb.sort()
        for i in range(len(filenames_hyper)):
            print("\n")
            print([filenames_hyper[i], filenames_rgb[i]])
            # load hyperspectral image
            mat =  h5py.File(filenames_hyper[i],'r')
            hyper = np.float32(np.array(mat['rad']))
            hyper = np.transpose(hyper, [0,2,1])
            hyper = normalize(hyper, max_val=4095., min_val=0.)
            mat.close()
            # load rgb image
            rgb =  cv2.imread(filenames_rgb[i])
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = np.transpose(rgb, [2,0,1])
            rgb = normalize(np.float32(rgb), max_val=255., min_val=0.)
            # add data
            data = np.concatenate((hyper,rgb), 0)
            h5f.create_dataset(str(test_num), data=data)
            test_num += 1
        h5f.close()
        print("\ntest set: # samples %d\n" % (test_num-1))

class HyperDataset(udata.Dataset):
    def __init__(self, crop_size=64, mode='train'):
        if (mode != 'train') & (mode != 'test'):
            raise Exception("Invalid mode!", mode)
        self.crop_size = crop_size
        self.mode = mode
        if self.mode == 'train':
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
        self.keys = list(h5f.keys())
        if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.mode == 'train':
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('test.h5', 'r')
        key = str(self.keys[index])
        data = np.array(h5f[key])
        data = torch.Tensor(data)
        # crop
        w = int(data.size()[1])
        h = int(data.size()[2])
        th, tw = self.crop_size, self.crop_size
        if w > tw or h > th:
            if self.mode == 'train': # random crop
                i = random.randint(0, w - tw)
                j = random.randint(0, h - th)
            else: # crop up left
                i = 0
                j = 0
            data = data[:,i:i+th,j:j+tw]
        h5f.close()
        return data[0:31,:,:], data[31:34,:,:]
