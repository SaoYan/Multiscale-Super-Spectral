import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def Loss_SAM(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    nom = torch.sum( torch.mul(im_true, im_fake), dim=1)
    denom1 = torch.sqrt( torch.sum( torch.pow(im_true,2), dim=1))
    denom2 = torch.sqrt( torch.sum( torch.pow(im_fake,2), dim=1))
    sam = torch.acos( torch.div(nom, torch.mul(denom1, denom2)))
    sam = torch.mul(torch.div(sam, np.pi), 180)
    sam = torch.div( torch.sum(sam), N*H*W)
    return sam

def batch_PSNR(im_true, im_fake, data_range):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W)
    psnr = 10. * torch.log((data_range**2)/err) / np.log(10.)
    return torch.mean(psnr)

def batch_SAM(im_true, im_fake):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clone().resize_(N, C, H*W)
    Ifake = im_fake.clone().resize_(N, C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=1).resize_(N, H*W)
    denom1 = torch.pow(Itrue,2).sum(dim=1).sqrt_().resize_(N, H*W)
    denom2 = torch.pow(Ifake,2).sum(dim=1).sqrt_().resize_(N, H*W)
    sam = torch.div(nom, torch.mul(denom1, denom2)).acos_().resize_(N, H*W)
    sam = sam / np.pi * 180
    sam = torch.sum(sam) / (N*H*W)
    return sam

def batch_RMSE(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sqrt_().sum(dim=1, keepdim=True).div_(C*H*W)
    return torch.mean(err)

def batch_RMSE_G(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C*H*W).sqrt_()
    return torch.mean(err)

def batch_rRMSE(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sqrt_().div_(Itrue).sum(dim=1, keepdim=True).div_(C*H*W)
    return torch.mean(err)

def batch_rRMSE_G(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    Ifake = im_fake.clamp(0.,1.).mul_(data_range).resize_(N, C*H*W)
    mse = nn.MSELoss(reduce=False)
    gt = torch.mean(Itrue).pow(2)
    err = mse(Itrue, Ifake).div_(gt).sum(dim=1, keepdim=True).div_(C*H*W).sqrt_()
    return torch.mean(err)

def plot_spectrum(real, fake):
    x = np.linspace(1, 31, 31, endpoint=True)
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    plot_real,  = ax.plot(x, real)
    plot_fake,  = ax.plot(x, fake)
    fig.legend((plot_real,plot_fake), ('real', 'fake'))
    canvas.draw()
    I = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
    I = I.reshape(canvas.get_width_height()[::-1]+(3,))
    I = np.transpose(I, [2,0,1])
    return np.float32(I)

def weights_init_kaimingUniform(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_uniform(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.uniform(m.weight.data, a=0, b=1)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.uniform(m.weight.data, a=0, b=1)
        nn.init.constant(m.bias.data, 0.0)

def weights_init_kaimingNormal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0.2, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)

def weights_init_xavierNormal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        nn.init.normal(m.weight.data, 0, 0.01)
        nn.init.constant(m.bias.data, 0.0)

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
