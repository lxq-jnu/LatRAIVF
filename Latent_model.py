import torch
from .base_model import BaseModel
from . import networks as networks
import numpy as np
import torch.nn.functional as F

class LatentModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='yuv')
        parser.set_defaults(where_add='input', nz=0)
        if is_train:
            parser.set_defaults(gan_mode='lsgan', lambda_l1=100.0)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
  
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['EnIR','EnVI','De', 'D']
        else:  # during test time, only load G
            self.model_names = ['EnIR','EnVI','De']
        # define networks (both generator and discriminator)



        self.netEnIR = networks.define_En_v0(opt.input_nc, 512, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        self.netEnVI = networks.define_En_v0(opt.input_nc, 512, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type, init_gain=opt.init_gain,
                                      gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        self.netDe = networks.define_De_v0(512, opt.output_nc, opt.ngf, netG=opt.netG,
                                           norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout,
                                           init_type=opt.init_type, init_gain=opt.init_gain,
                                           gpu_ids=self.gpu_ids, where_add=opt.where_add, upsample=opt.upsample)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc


            self.netD = networks.define_D(opt.input_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            #if self.isTrain and not opt.no_ganFeat_loss:
            #    self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            #else:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            #self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionSSIM = networks.SSIM()  # SSIM 结构相似性损失作为图像的重构损失
            self.criterionL1 = torch.nn.L1Loss()
            self.lap = LaplacianConv()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionTV = networks.TVLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_EnIR = torch.optim.Adam(self.netEnIR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_EnVI = torch.optim.Adam(self.netEnVI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_De = torch.optim.Adam(self.netDe.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_EnIR)
            self.optimizers.append(self.optimizer_EnVI)
            self.optimizers.append(self.optimizer_De)
            self.optimizers.append(self.optimizer_D)

            if self.fuse_mode == "adapt":
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_F)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if 'F' in input.keys():
            self.real_F = input['F'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        with torch.no_grad():
            # concat_AB = torch.cat([self.real_A,self.real_B],1)
            self.cat_dict_AB = {}
            self.cat_A, self.lat_A = self.netEnIR(self.real_A)
            self.cat_B, self.lat_B = self.netEnVI(self.real_B)
            for key in self.cat_A:

                self.cat_dict_AB[key] = torch.cat([self.cat_A[key], self.cat_B[key]], 1)

            self.fake_lat = torch.max(self.lat_A, self.lat_B)
            self.fake_F = self.netDe(self.fake_lat,self.cat_dict_AB)

            return self.real_A, self.fake_F, self.real_B



    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.cat_A,self.lat_A = self.netEnIR(self.real_A)
        self.cat_B, self.lat_B = self.netEnVI(self.real_B)

        self.cat_dict_AB = {}
        for key in self.cat_A:
            self.cat_dict_AB[key] = torch.cat([self.cat_A[key],self.cat_B[key]],1)

        self.fake_lat = torch.max(self.lat_A, self.lat_B)

        self.fake_F = self.netDe(self.fake_lat,self.cat_dict_AB)


        _,self.fake_lat_ir = self.netEnIR(self.fake_F)
        _, self.fake_lat_vi = self.netEnVI(self.fake_F)
        self.fake_lat2 = torch.max(self.fake_lat_ir,self.fake_lat_vi)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        pred_fake = self.netD(self.fake_F.detach())

        self.loss_D_fake,_  = self.criterionGAN(pred_fake, False)

        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(self.real_F)
        self.loss_D_real,_  = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real)



        #self.loss_D.backward()

    def backward_GE(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(self.fake_F)
        self.loss_G_GAN,_ = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_F, self.real_F) * self.opt.lambda_L1

        self.loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            if self.opt.input_nc == 1:
                cat_fake = torch.cat([self.fake_F,self.fake_F,self.fake_F], 1)
                cat_real = torch.cat([self.real_F, self.real_F, self.real_F], 1)
            else:
                cat_fake = self.fake_F
                cat_real = self.real_F
            self.loss_G_VGG = self.criterionVGG(cat_fake, cat_real) * self.opt.lambda_feat

        self.loss_SSIM = 0
        if self.opt.lambda_SSIM > 0.0:
            self.loss_SSIM = (1 - self.criterionSSIM(self.fake_F, self.real_F)) * self.opt.lambda_SSIM


        self.loss_grad = 0
        if self.opt.lambda_grad > 0.0:

            self.realF_grad = self.lap(self.real_F)
            self.fakeF_grad = self.lap(self.fake_F)
            self.loss_grad = self.criterionL1(self.fakeF_grad, self.realF_grad) * self.opt.lambda_grad

        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = (self.criterionMSE(self.fake_lat_ir, self.lat_A)+self.criterionMSE(self.fake_lat_vi, self.lat_B)) * self.opt.lambda_z
        else:
            self.loss_z_L1 = 0


        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1+self.loss_G_VGG+ self.loss_z_L1



    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        #update GE
        #self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()
        self.loss_D.backward()
        self.optimizer_D.step()  # update D's weights

        #self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_EnIR.zero_grad()  # set G's gradients to zero
        self.optimizer_EnVI.zero_grad()
        self.optimizer_De.zero_grad()


        self.backward_GE()  # calculate graidents for G
        self.loss_G.backward()
        self.optimizer_EnIR.step()  # udpate G's weights
        self.optimizer_EnVI.step()
        self.optimizer_De.step()




class LaplacianConv(torch.nn.Module):

    def __init__(self, channels=1):
        super().__init__()

        if channels ==3:
            self.filter = torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1,bias=False,groups=3)
        else:
            self.filter = torch.nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1,bias=False,groups=1)
        print(self.filter.weight.size())
        self.channels = channels
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()  

        if channels == 3:
            kernel = torch.cat([kernel,kernel,kernel],0)
            #kernel = torch.cat([kernel,kernel,kernel],1)

        self.filter.weight = torch.nn.Parameter(kernel, requires_grad=False)

    def __call__(self, x):

        x = self.filter(x)
        return x
