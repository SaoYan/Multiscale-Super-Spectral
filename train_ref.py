import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as udata
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataset import (HyperDataset, process_data)
from model_ref import Model_Ref # Fully Conv DenseNet
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

parser = argparse.ArgumentParser(description="SpectralSR-Ref")
parser.add_argument("--preprocess", type=bool, default=False, help="whether to run prepare_data")
parser.add_argument("--batchSize", type=int, default=128, help="batch size")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--milestone", type=int, default=50, help="when to decay lr")
parser.add_argument("--lr", type=float, default=2e-3, help="initial learning rate")
parser.add_argument("--pdrop", type=float, default=0.5, help="dropout rate")
parser.add_argument("--outf", type=str, default="logs", help='path log files')

opt = parser.parse_args()

def main():
    ## network architecture
    rgb_features = 3
    pre_features = 64
    hyper_features = 31
    growth_rate = 16
    negative_slope = 0.2
    ## optimization
    beta1 = 0.9
    beta2 = 0.999
    ## load dataset
    print("\nloading dataset ...\n")
    trainDataset = HyperDataset(crop_size=64, mode='train')
    trainLoader = udata.DataLoader(trainDataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    testDataset = HyperDataset(crop_size=1280, mode='test')
    ## build model
    print("\nbuilding models ...\n")
    net = Model_Ref(
        input_features = rgb_features,
        pre_features = pre_features,
        output_features = hyper_features,
        db_growth_rate = growth_rate,
        negative_slope = negative_slope,
        p_drop = opt.pdrop
    )
    net.apply(weights_init_kaimingUniform)
    criterion = nn.MSELoss()
    # move to GPU
    device_ids = [0,1,2,3,4,5,6,7]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # optimizers
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6, betas=(beta1, beta2))
    ## begin training
    step = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        # set learning rate
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print("\nepoch %d learning rate %f\n" % (epoch, current_lr))
        # run for one epoch
        for i, data in enumerate(trainLoader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            real_hyper, real_rgb = data
            H = real_hyper.size(2)
            W = real_hyper.size(3)
            real_hyper, real_rgb = Variable(real_hyper.cuda()), Variable(real_rgb.cuda())
            # train
            fake_hyper = model.forward(real_rgb)
            loss = criterion(fake_hyper, real_hyper)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                # result
                model.eval()
                with torch.no_grad():
                    fake_hyper = model.forward(real_rgb)
                RMSE = batch_RMSE(real_hyper, fake_hyper)
                RMSE_G = batch_RMSE_G(real_hyper, fake_hyper)
                rRMSE = batch_rRMSE(real_hyper, fake_hyper)
                rRMSE_G = batch_rRMSE_G(real_hyper, fake_hyper)
                SAM = batch_SAM(real_hyper, fake_hyper)
                print("[epoch %d][%d/%d] RMSE: %.4f RMSE_G: %.4f rRMSE: %.4f rRMSE_G: %.4f SAM: %.4f"
                    % (epoch, i, len(trainLoader), RMSE.item(), RMSE_G.item(), rRMSE.item(), rRMSE_G.item(), SAM.item()))
                # Log the scalar values
                writer.add_scalar('RMSE', RMSE.item(), step)
                writer.add_scalar('RMSE_G', RMSE_G.item(), step)
                writer.add_scalar('rRMSE', rRMSE.item(), step)
                writer.add_scalar('rRMSE_G', rRMSE_G.item(), step)
                writer.add_scalar('SAM', SAM.item(), step)
            if (i == 0) | (i == 1000):
                # validate
                model.eval()
                print("\ncomputing results on validation set ...\n")
                num = len(testDataset)
                average_RMSE = 0.
                average_RMSE_G = 0.
                average_rRMSE = 0.
                average_rRMSE_G = 0.
                average_SAM  = 0.
                for k in range(num):
                    # data
                    real_hyper, real_rgb = testDataset[k]
                    real_hyper = torch.unsqueeze(real_hyper, 0)
                    real_rgb = torch.unsqueeze(real_rgb, 0)
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
                writer.add_scalar('RMSE_val', average_RMSE/num, step)
                writer.add_scalar('RMSE_G_val', average_RMSE_G/num, step)
                writer.add_scalar('rRMSE_val', average_rRMSE/num, step)
                writer.add_scalar('rRMSE_G_val', average_rRMSE_G/num, step)
                writer.add_scalar('SAM_val', average_SAM/num, step)
                print("[epoch %d][%d/%d] validation:\nRMSE: %.4f RMSE_G: %.4f rRMSE: %.4f rRMSE_G: %.4f SAM: %.4f\n"
                    % (epoch, i, len(trainLoader), average_RMSE/num, average_RMSE_G/num, average_rRMSE/num, average_rRMSE_G/num, average_SAM/num))
            step += 1
        ## the end of each epoch
        model.eval()
        # plot spectrum
        print("\nplotting spectrum ...\n")
        with torch.no_grad():
            fake_hyper = model.forward(real_rgb)
        real_spectrum = real_hyper.data.cpu().numpy()[0,:,int(H/2),int(W/2)]
        fake_spectrum = fake_hyper.data.cpu().numpy()[0,:,int(H/2),int(W/2)]
        I_spectrum = plot_spectrum(real_spectrum, fake_spectrum)
        writer.add_image('spectrum', torch.Tensor(I_spectrum), epoch)
        # images
        print("\nadding images ...\n")
        I_rgb = utils.make_grid(real_rgb.data[0:16,:,:,:].clamp(0.,1.), nrow=4, normalize=True, scale_each=True)
        I_real = utils.make_grid(real_hyper.data[0:16,0:3,:,:].clamp(0.,1.), nrow=4, normalize=True, scale_each=True)
        I_fake = utils.make_grid(fake_hyper.data[0:16,0:3,:,:].clamp(0.,1.), nrow=4, normalize=True, scale_each=True)
        writer.add_image('rgb', I_rgb, epoch)
        writer.add_image('real', I_real, epoch)
        writer.add_image('fake', I_fake, epoch)
        # save model
        if epoch >= opt.epochs - 10:
            torch.save(model.state_dict(), os.path.join(opt.outf, 'net_%03d.pth'%epoch))

if __name__ == "__main__":
    if opt.preprocess:
        process_data(patch_size=64, stride=40, path='data', mode='train')
        process_data(patch_size=None, stride=None, path='data', mode='test')
    main()
