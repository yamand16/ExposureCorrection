import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance', gpu_ids=[], is_shortcut=False):
    norm_layer = get_norm_layer(norm_type=norm)         
    
    netG = GlobalGenerator(input_nc=input_nc, output_nc=output_nc, ngf=ngf, n_downsampling=n_downsample_global, n_blocks=n_blocks_global, norm_layer=norm_layer, is_shortcut=is_shortcut)

    print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[], spectral_norm=False, dropout_=False, no_lsgan=False):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat, spectral_norm, dropout_, no_lsgan=no_lsgan)   
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            print("BCE loss")
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

class InpaintingLoss(nn.Module):
    def __init__(self, inverse_inpainting=False):
        super(InpaintingLoss, self).__init__()
        self.loss_func = torch.nn.L1Loss(reduction='sum')
        self.inverse = inverse_inpainting
        
    def forward(self, x, y, seg, number_of_pixels):
        loss = 0.0
        # TODO
        if self.inverse:
            seg_inverse = 1.0 - seg
            x_seg_inv = np.multiply(x.cpu().detach(), seg_inverse.cpu().detach()).cuda()
            y_seg_inv = np.multiply(y.cpu().detach(), seg_inverse.cpu().detach()).cuda()
            pre_loss_inv = self.loss_func(x_seg_inv, y_seg_inv)
            all_number_of_pixels = torch.sum(number_of_pixels)
            loss = pre_loss_inv / (x.shape[2]*x.shape[3]*x.shape[0] - all_number_of_pixels)

        else:
            x_seg = np.multiply(x.cpu().detach(), seg.cpu().detach()).cuda()
            y_seg = np.multiply(y.cpu().detach(), seg.cpu().detach()).cuda()
            pre_loss = self.loss_func(x_seg, y_seg)
            all_number_of_pixels = torch.sum(number_of_pixels)
            loss = pre_loss / all_number_of_pixels
        # END TODO
        return loss

class StyleDiscriminatorLoss(nn.Module):
    def __init__(self, gpu_ids, average_=True, tensor=torch.FloatTensor):
        super(StyleDiscriminatorLoss, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.average_ = average_
        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, predicted_styles, GTs):
        loss = 0.0
        #print(len(predicted_styles))
        for i in predicted_styles:
            for j in i:
                #target_tensor = self.get_target_tensor(i[-1], GTs)
                #loss += self.loss_func(i[-1], target_tensor)
                target_tensor = self.get_target_tensor(j, GTs)
                loss += self.loss_func(j, target_tensor)
        
        if self.average_:
            loss /= (len(predicted_styles) * len(predicted_styles[0]))
        else:
            loss /= len(predicted_styles)

        return loss
##############################################################################
# Generator
##############################################################################

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', activation=nn.ReLU(True), is_shortcut=False):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()

        self.is_shortcut = is_shortcut        
        self.n_downsampling = n_downsampling      
        self.ngf = ngf
        self.activation = activation

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]
            
        ### upsample
        if self.is_shortcut:
            ### Residual block
            res_model = []
            mult = 2**n_downsampling
            for i in range(n_blocks):
                res_model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            self.res_model = nn.Sequential(*res_model)

            decoder_network = []
            mult_input = int(ngf * mult)
            for i in range(n_downsampling):
                mult_output = (2 ** (n_downsampling - i)) / 2
                decoder_network += [nn.ConvTranspose2d(mult_input, int(ngf * mult_output), kernel_size=3, stride=2, padding=1, output_padding=1),    norm_layer(int(ngf * mult / 2)), activation] 
                mult_input = ngf * (int(mult_output) * 2)
            decoder_network += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]   
            self.decoder_network = nn.Sequential(*decoder_network)
        else:
            ### Residual block
            mult = 2**n_downsampling
            for i in range(n_blocks):
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            
            for i in range(n_downsampling):
                mult = 2**(n_downsampling - i)
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer(int(ngf * mult / 2)), activation]

            model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input=None):
        if self.is_shortcut:
            outputs = [input]
            res_concat = []
            shortcut_dim = []
            for i in range(1, self.n_downsampling):
                shortcut_dim.append(self.ngf * int(2 ** i))

            for i in range(len(self.model.model)):
                outputs.append(self.model.model[i](outputs[-1]))
                if self.model.model[i].__class__.__name__ is self.activation.__class__.__name__ and outputs[-1].shape in shortcut_dim:
                    res_concat.append(outputs[-1])

            res_output = [outputs[-1]]
            for i in range(len(self.res_model.res_model)):
                res_output.append(self.res_model.res_model[i](res_output[-1]))

            decoding_output = [res_output[-1]]
            index_res = len(res_concat) - 1
            for i in range(len(self.decoder_network.decoder_network)):
                x = self.decoder_network.decoder_network[i](decoding_output[-1])
                if self.decoder_network.decoder_network[i].__class__.__name__ is self.activation.__class__.__name__ and x.shape[1] in shortcut_dim:
                    decoding_output.append(torch.concat((x, res_concat[index_res]), dim=1))
                    index_res -= 1
                else:
                    decoding_output.append(x)
            return decoding_output[-1]
        else:
            return self.model(input)    

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False, spectral_norm=False, dropout_=False, no_lsgan=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.dropout_ = dropout_
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat, spectral_norm, dropout_, no_lsgan)
            if getIntermFeat:                                  
                if self.dropout_:
                    for j in range((n_layers*2)+3):
                        setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))         
                else:
                    if self.self_attention:
                        for j in range(n_layers+4):
                            setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))         
                    else:
                        for j in range(n_layers+2):
                            setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                           
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        convenient_outputs = []
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))

            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                if self.dropout_:
                    model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range((self.n_layers*2)+3)]
                else:
                    model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result_ = self.singleD_forward(model, input_downsampled)
            result.append(result_)
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False, spectral_norm=False, dropout_=False, no_lsgan=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.no_lsgan = no_lsgan

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        if spectral_norm:
            sequence = [[nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]]
        else:
            sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        if dropout_:
            sequence += [[nn.Dropout2d(p=0.25)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            if spectral_norm:
                sequence += [[
                    nn.utils.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
            else:
                sequence += [[
                    nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(nf), nn.LeakyReLU(0.2, True)
                ]]
            if dropout_:
                sequence += [[nn.Dropout2d(p=0.25)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        if spectral_norm:
            sequence += [[
                nn.utils.spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]
        else:
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
                norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        if dropout_:
            sequence += [[nn.Dropout2d(p=0.25)]]

        if no_lsgan:
            if spectral_norm:
                sequence += [[nn.utils.spectral_norm(nn.Conv2d(nf, nf, kernel_size=kw, stride=2, padding=padw)), nn.Flatten(), nn.Linear(nf*18*18, 1), nn.Sigmoid()]]
            else:
                sequence += [[nn.Conv2d(nf, nf, kernel_size=kw, stride=2, padding=padw), nn.Flatten(), nn.Linear(nf*18*18, 1)]]
        else:
            if spectral_norm:
                sequence += [[nn.utils.spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))]]
            else:
                sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            if self.dropout_:
                upper_bound = self.n_layers*2 + 3
            else:
                upper_bound = self.n_layers+2
            for n in range(upper_bound):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            out = self.model(input)
            return out      

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
