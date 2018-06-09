import os
import argparse
import cv2
import torch
import numpy as np
import scipy.io as scio
import torch.nn as nn
import torch.utils.data as udata
import torchvision.utils as utils
from torch.autograd import Variable
from dataset import (process_data, HyperDataset)
from tensorboardX import SummaryWriter
from model import Model # my model
from model_ref import Model_Ref # Fully Conv DenseNet
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser(description="LearnedSSR")
parser.add_argument("--preprocess", type=bool, default=False, help="Whether to run prepare_data")
parser.add_argument("--logs", type=str, default="TestLog", help='path to training log files')
parser.add_argument("--size", type=int, default=1280, help="crop size of test sample")
parser.add_argument("--model", type=str, default='Model', help="Model/Ref")
parser.add_argument("--dropout", type=int, default='0', help="0 for 0; 2 for 0.2; 5 for 0.5")

opt = parser.parse_args()

def main():
    ## network architecture
    rgb_features = 3
    pre_features = 64
    hyper_features = 31
    growth_rate = 16
    negative_slope = 0.2
    ## load data
    print("loading dataset ...\n")
    testDataset = HyperDataset(crop_size=opt.size, mode='test')
    ## build model
    print("building models ...\n")
    if opt.model == 'Model':
        print("Our model, Dropout rate 0.%d\n" % opt.dropout)
        net = Model(
            input_features = rgb_features,
            output_features = hyper_features,
            negative_slope = negative_slope,
            p_drop = 0
        )
    elif opt.model == 'Ref':
        print("Reference model FC-DenseNet, Dropout rate 0.%d\n" % opt.dropout)
        net = Model_Ref(
            input_features = rgb_features,
            pre_features = pre_features,
            output_features = hyper_features,
            db_growth_rate = growth_rate,
            negative_slope = negative_slope,
            p_drop = 0
        )
    else:
        raise Exception("Invalid model name!", opt.model)
    # move to GPU
    device_ids = [0,1,2,3,4,5,6,7]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logs, 'net_%s_%02d.pth'%(opt.model,opt.dropout) )))
    model.eval()
    ## testing
    num = len(testDataset)
    average_RMSE = 0.
    average_RMSE_G = 0.
    average_rRMSE = 0.
    average_rRMSE_G = 0.
    average_SAM  = 0.
    criterion = nn.MSELoss()
    criterion.cuda()
    # recording results
    im_rgb = dict()
    im_hyper = dict()
    im_hyper_fake = dict()
    for i in range(num):
        # data
        real_hyper, real_rgb = testDataset[i]
        real_hyper = torch.unsqueeze(real_hyper, 0)
        real_rgb = torch.unsqueeze(real_rgb, 0)
        H = real_hyper.size(2)
        W = real_hyper.size(3)
        real_hyper, real_rgb = Variable(real_hyper.cuda()), Variable(real_rgb.cuda())
        # forward
        with torch.no_grad():
            fake_hyper = model.forward(real_rgb)
        # metrics
        RMSE = batch_RMSE(real_hyper, fake_hyper)
        RMSE_G = batch_RMSE_G(real_hyper, fake_hyper)
        rRMSE = batch_rRMSE(real_hyper, fake_hyper)
        rRMSE_G = batch_rRMSE_G(real_hyper, fake_hyper)
        SAM = batch_SAM(real_hyper, fake_hyper)
        average_RMSE    += RMSE.item()
        average_RMSE_G  += RMSE_G.item()
        average_rRMSE   += rRMSE.item()
        average_rRMSE_G += rRMSE_G.item()
        average_SAM     += SAM.item()
        print("[%d/%d] RMSE: %.4f RMSE_G: %.4f rRMSE: %.4f rRMSE_G: %.4f SAM: %.4f"
            % (i+1, num, RMSE.item(), RMSE_G.item(), rRMSE.item(), rRMSE_G.item(), SAM.item()))
        # images
        print("adding images ...\n")
        I_rgb = real_rgb.data.cpu().numpy().squeeze()
        I_hyper = real_hyper.data.cpu().numpy().squeeze()
        I_hyper_fake = fake_hyper.data.cpu().numpy().squeeze()
        im_rgb['rgb_%d'%i] = I_rgb
        im_hyper['hyper_%d'%i] = I_hyper
        im_hyper_fake['hyper_fake_%d'%i] = I_hyper_fake

    print("\naverage RMSE: %.4f" % (average_RMSE/num))
    print("average RMSE_G: %.4f" % (average_RMSE_G/num))
    print("\naverage rRMSE: %.4f" % (average_rRMSE/num))
    print("average rRMSE_G: %.4f" % (average_rRMSE_G/num))
    print("\naverage SAM: %.4f" % (average_SAM/num))

    print("\nsaving matlab files ...\n")
    scio.savemat(os.path.join(opt.logs, 'im_rgb.mat'), im_rgb)
    scio.savemat(os.path.join(opt.logs, 'im_hyper.mat'), im_hyper)
    scio.savemat(os.path.join(opt.logs, 'im_hyper_fake.mat'), im_hyper_fake)

if __name__ == "__main__":
    if opt.preprocess:
        process_data(patch_size=None, stride=None, path='NTIRE2018', mode='test')
    main()
