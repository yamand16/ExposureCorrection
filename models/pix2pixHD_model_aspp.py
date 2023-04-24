import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
import torch.nn as nn
from .base_model import BaseModel
import torch.nn.functional as F
import time
from . import networks_mine as networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_l1_image_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, use_l1_image_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_image, d_real, d_fake):
            return [l for (l,f) in zip((g_gan, g_gan_feat, g_vgg, g_image, d_real, d_fake), flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        input_nc = opt.label_nc
        
        ##### define networks        
        # Generator network
        netG_input_nc = input_nc     
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, 
                                      opt.n_downsample_global, opt.n_blocks_global, 
                                      opt.norm, gpu_ids=self.gpu_ids, is_shortcut=opt.is_shortcut)    

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.single_input_D:
                netD_input_nc = input_nc
            else:
                netD_input_nc = input_nc + opt.output_nc
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids, spectral_norm=opt.spectral_normalization_D, dropout_=opt.dropout_D, no_lsgan=opt.no_lsgan)

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)     
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
                
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.old_lr_D = opt.lr_D

            # define loss functions
            
            self.loss_filter = self.init_loss_filter((not opt.no_ganFeat_loss), (not opt.no_vgg_loss), opt.l1_image_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)    
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_Image','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr_D, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            
    def encode_input(self, label_map, real_image=None, infer=False):             
        input_label = label_map.data.cuda()
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None and self.opt.isTrain:
            real_image = Variable(real_image.data.cuda())

        return input_label, real_image

    def discriminate(self, input_label, test_image, use_pool=False, is_single_input=False):
        if is_single_input:
            input_concat = test_image
        else:
            input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
            
    def forward(self, label, image, infer=False, number_of_pixel=None, min_row=None, max_row=None, min_col=None, max_col=None):
        # Encode Inputs
        input_concat, real_image = self.encode_input(label, image)  

        # Fake Generation    
        fake_image = self.netG.forward(input_concat)

        disc_fake = fake_image.copy()
        disc_input = input_concat.copy()
        disc_real = real_image.copy()

        pred_fake_pool = self.discriminate(disc_input, disc_fake, use_pool=True, is_single_input=self.opt.single_input_D)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        
        # Real Detection and Loss        
        pred_real = self.discriminate(disc_input, disc_real, is_single_input=self.opt.single_input_D)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)  
        loss_G_GAN = 0      
        if self.opt.single_input_D:
            pred_fake = self.netD.forward(disc_fake)
        else:
            pred_fake = self.netD.forward(torch.cat((disc_input, disc_fake), dim=1))  
        loss_G_GAN = self.criterionGAN(pred_fake, True)        
                
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
                
        # L1 image comparison
        loss_G_Image = 0
        if self.opt.l1_image_loss:# and (not self.opt.is_style_encoder):
            loss_G_Image = self.criterionL1Image(fake_image, real_image) * self.opt.l1_image_loss_coef

        return [ self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_Image, loss_D_real, loss_D_fake), None if not infer else fake_image, None]

    def inference(self, label, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_concat, real_image = self.encode_input(Variable(label), image, infer=True)      
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        
        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        
    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        

        lrd_D = self.opt.lr_D / self.opt.niter_decay
        lr_D = self.old_lr_D - lrd_D

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr_D
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate G: %f -> %f' % (self.old_lr, lr))
            print('update learning rate D: %f -> %f' % (self.old_lr_D, lr_D))
        self.old_lr = lr
        self.old_lr_D = lr_D

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp, inp2 = None, is_style_encoder=False, use_segmentation_map_as_input=False, seg_map_input=None):
        label, inst = inp
        return self.inference(label, inst)

        
