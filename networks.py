import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from spectral_normalization import SpectralNorm



###############################################################################
# Helper functions
###############################################################################



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def init_net_test(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_D(input_nc, ndf, netD, norm='batch', nl='lrelu', init_type='xavier', init_gain=0.02, num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    if netD == 'basic_128':
        net = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_256':
        net = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer, nl_layer=nl_layer)
    elif netD == 'basic_128_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer, num_D=num_Ds)
    elif netD == 'basic_256_multi':
        net = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, num_D=num_Ds)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2 ** i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw,
                                           stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                       kernel_size=kw, stride=2, padding=padw)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
            #Self_Attn(ndf * nf_mult,"lrelu")#新增注意力机制
        ]



        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1,
                                            kernel_size=kw, stride=1, padding=padw))]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class D_NLayers(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(D_NLayers, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


'''
pix2pix HD
'''


# Defines the PatchGAN discriminator with the specified arguments.


##############################################################################
# Classes
##############################################################################
class RecLoss(nn.Module):
    def __init__(self, use_L2=True):
        super(RecLoss, self).__init__()
        self.use_L2 = use_L2

    def __call__(self, input, target, batch_mean=True):
        if self.use_L2:
            diff = (input - target) ** 2
        else:
            diff = torch.abs(input - target)
        if batch_mean:
            return torch.mean(diff)
        else:
            return torch.mean(torch.mean(torch.mean(diff, dim=1), dim=2), dim=3)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor list) - - tpyically the prediction output from a discriminator; supports multi Ds.
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        all_losses = []
        for prediction in predictions:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck



def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  SpectralNorm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    elif upsample == 'nearest':
        upconv = [nn.Upsample(scale_factor=2, mode='nearest'),
                  nn.ReflectionPad2d(1),
                  SpectralNorm(nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0))]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv




def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)




###TV_loss#
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# 计算SSIM
# 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
# 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
# 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
# @torchsnooper.snoop()

import torch
import torch.nn.functional as F
from math import exp
import numpy as np


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 协方差期望公式：sigma_x=E（X^2）-（EX）^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # 协方差期望公式：sigma_xy=E(XY）-（EX）*（EY）
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # mm是计算两个矩阵/向量的乘积，（m，n）*(n,p)=(m,p)
    # bmm多了一维b，即可以进行批矩阵/向量的乘积
    # matual可以进行高维张量乘法，但是非矩阵的维度会被广播（前提是能广播的情况下）
    # i.e.（j,1,n,m）*(k,m,p)=(j,k,n,p)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # 通道扩展
    return window


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 3 channel for SSIM
        self.channel = channel
        self.window = create_window(window_size, channel)

    # @torchsnooper.snoop()
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window.to(img1.device)
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


# Define perceptual loss

from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


#################################################模型修改###############################################
###############V0####################

class Encoder_v0(nn.Module):
    def __init__(self, input_nc, outer_nc, ngf=64,
                 norm_layer=None, nl_layer=None, upsample='basic', padding_type='reflect'):
        super().__init__()
        p = 0
        downconv = []

        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
            pad = [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
            pad = [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
            pad = None
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [SpectralNorm(nn.Conv2d(input_nc, 64,
                                            kernel_size=4, stride=2, padding=p))]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)

        down1 = pad + [SpectralNorm(nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=p))]
        self.down1 = nn.Sequential(*down1)
        # self.down1 = nn.Sequential(*downconv)

        down2 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf, ngf * 2,
                                                           kernel_size=4, stride=2, padding=p))] + [norm_layer(ngf * 2)]

        self.down2 = nn.Sequential(*down2)

        down3 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf * 2, ngf * 4,
                                                           kernel_size=4, stride=2, padding=p))] + [norm_layer(ngf * 4)]

        self.down3 = nn.Sequential(*down3)

        down4 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf * 4, ngf * 8,
                                                           kernel_size=4, stride=2, padding=p))] + [norm_layer(ngf * 8)]

        self.down4 = nn.Sequential(*down4)

        down5 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf * 8, ngf * 8,
                                                           kernel_size=4, stride=2, padding=p))] + [norm_layer(ngf * 8)]

        self.down5 = nn.Sequential(*down5)

        down6 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf * 8, ngf * 8,
                                                           kernel_size=4, stride=2, padding=p))] + [norm_layer(ngf * 8)]

        self.down6 = nn.Sequential(*down6)

        down7 = [downrelu] + pad + [SpectralNorm(nn.Conv2d(ngf * 8, outer_nc,
                                                           kernel_size=4, stride=2, padding=p))]
        self.down7 = nn.Sequential(*down7)

    def forward(self, x):

        x1 = self.down1(x)

        x2 = self.down2(x1)

        x3 = self.down3(x2)

        x4 = self.down4(x3)

        x5 = self.down5(x4)

        x6 = self.down6(x5)

        latent = self.down7(x6)

        cat_dict = {"cat_1": x1, "cat_2": x2, "cat_3": x3, "cat_4": x4, "cat_5": x5, "cat_6": x6}

        return cat_dict, latent


class Decoder_v0(nn.Module):
    def __init__(self, input_nc, outer_nc, ngf=64,
                 norm_layer=None, nl_layer=None, upsample='basic', padding_type='reflect'):
        super().__init__()

        pad = [nn.ReflectionPad2d(1)]
        p = 0
        uprelu = nl_layer()

        upconv = upsampleLayer(
            ngf * 3, outer_nc, upsample=upsample, padding_type=padding_type)
        upconv = pad + [SpectralNorm(nn.Conv2d(ngf * 3, outer_nc, kernel_size=3, stride=1, padding=p))]

        # up1 = [uprelu] + upconv + [nn.Tanh()]
        up1 = [uprelu] + upconv + [nn.Tanh()]
        self.up1 = nn.Sequential(*up1)

        up2 = [uprelu] + upsampleLayer(
            ngf * 2 * 3, ngf, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf)]
        self.up2 = nn.Sequential(*up2)

        up3 = [uprelu] + upsampleLayer(
            ngf * 4 * 3, ngf * 2, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf * 2)]
        self.up3 = nn.Sequential(*up3)

        up4 = [uprelu] + upsampleLayer(
            ngf * 8 * 3, ngf * 4, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf * 4)]
        self.up4 = nn.Sequential(*up4)

        up5 = [uprelu] + upsampleLayer(
            ngf * 8 * 3, ngf * 8, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf * 8)]
        self.up5 = nn.Sequential(*up5)

        up6 = [uprelu] + upsampleLayer(
            ngf * 8 * 3, ngf * 8, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf * 8)]
        self.up6 = nn.Sequential(*up6)

        up7 = [uprelu] + upsampleLayer(
            input_nc, ngf * 8, upsample=upsample, padding_type=padding_type) + [norm_layer(ngf * 8)]
        self.up7 = nn.Sequential(*up7)

    def forward(self, x, cat_dict):
        x = self.up7(x)

        x = self.up6(torch.cat([x, cat_dict["cat_6"]], 1))

        x = self.up5(torch.cat([x, cat_dict["cat_5"]], 1))

        x = self.up4(torch.cat([x, cat_dict["cat_4"]], 1))

        x = self.up3(torch.cat([x, cat_dict["cat_3"]], 1))

        x = self.up2(torch.cat([x, cat_dict["cat_2"]], 1))

        x = self.up1(torch.cat([x, cat_dict["cat_1"]], 1))

        return x


def define_En_v0(input_nc, output_nc, ngf, netG='unet_128', norm='instance', nl='lrelu',
                 use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='nearest',
                 padding_type='reflect'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    # if nz == 0:
    #    where_add = 'input'

    net = Encoder_v0(input_nc, output_nc, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                     upsample=upsample, padding_type=padding_type)

    return init_net_test(net, init_type, init_gain)  # 返回经过初始化的网络


def define_De_v0(input_nc, output_nc, ngf, netG='unet_128', norm='instance', nl='lrelu',
                 use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[], where_add='input', upsample='nearest',
                 padding_type='reflect'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    # if nz == 0:
    #    where_add = 'input'

    net = Decoder_v0(input_nc, output_nc, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                     upsample=upsample, padding_type=padding_type)

    return init_net_test(net, init_type, init_gain)  # 返回经过初始化的网络


