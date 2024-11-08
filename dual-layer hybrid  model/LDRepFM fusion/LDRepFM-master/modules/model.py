from collections import OrderedDict

import torch.nn as nn
import numpy as np
import torch
import copy
import torch.nn.functional as F
class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()




#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True



def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


class LseRepFusNet(nn.Module):

    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(LseRepFusNet, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        # self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.in_planes = 32

        self.encoder0 = RepVGGBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        # self.encoder1 = RepVGGBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        # self.encoder2 = RepVGGBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.encoder1 = RepVGGBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.encoder2 = self._make_our_stage()



        self.suppleConv0 = ConvBnLeakyRelu2d(1, 64)
        # self.suppleConv1 = ConvRelu2d(48, 96)
        self.decoder0 = ConvBNRelu2d(128, 64)
        self.decoder1 = ConvBnLeakyRelu2d(64, 32)
        self.decoder2 = ConvBnTanh2d(32, 1)

        # self.suppleConv0 = ConvBnLeakyRelu2d(1, 48)
        # self.suppleConv1 = ConvBnLeakyRelu2d(48, 96)
        # self.decoder0 = ConvBNRelu2d(192, 96)
        # self.decoder1 = ConvBnLeakyRelu2d(96, 48)
        # self.decoder2 = ConvBnTanh2d(48, 1)

        # self.encoder0 = RepVGGBlock(in_channels=1, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        # self.cur_layer_idx = 1
        # self.encoder1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        # self.encoder2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=1)

        # self.encoder0_vi = copy.deepcopy(self.encoder0)
        # self.encoder1_vi = copy.deepcopy(self.encoder1)
        # self.encoder2_vi = copy.deepcopy(self.encoder2)

        # self.decoder0 = ConvBnLeakyRelu2d(192, 96)
        # self.decoder0 = ConvBnLeakyBNRelu2d(192, 96)
        # self.decoder1 = ConvBnLeakyRelu2d(96, 48)
        # self.decoder2 = ConvBnLeakyRelu2d(48, 24)
        # self.decoder2 = ConvBnLeakyBNRelu2d(48, 24)
        # self.decoder3 = ConvBnTanh2d(48, 1)

        # self.PA = Position_Attention(self.deploy, self.use_se)
        # self.con = ConvBnLeakyRelu2d(192, 96)
        # self.con2 = ConvBnLeakyRelu2d(1, 48)
        # self.con3 = ConvBnLeakyRelu2d(48, 96)
        #
        # self.sobel = Sobelxy(96)
        # self.sobelConv = ConvBnLeakyRelu2d(96, 96)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def _make_our_stage(self):
        blocks = []
        blocks.append(RepVGGBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, deploy=self.deploy,use_se=self.use_se))
        for stride in range(2):
            blocks.append(RepVGGBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
                                    deploy=self.deploy, use_se=self.use_se))
        return nn.Sequential(*blocks)


    def attention_P(self, features):
        return

    def forward_bak(self, ir, vi):
        out = self.encoder0(ir)
        out = self.encoder1(out)
        ir_f = self.encoder2(out)
        ir_f_p = ir_f * self.PA(ir_f)
        ir_f = torch.cat([ir_f, ir_f_p], dim=1)
        ir_f = self.con(ir_f)
        ir_f_grad = self.sobel(ir_f)
        ir_f = ir_f + ir_f_grad
        ir_f = self.sobelConv(ir_f)

        out = self.encoder0(vi)
        out = self.encoder1(out)
        vi_f = self.encoder2(out)
        vi_f_grad = self.sobel(vi_f)
        vi_f = vi_f + vi_f_grad
        vi_f = self.sobelConv(vi_f)

        out = torch.cat([vi_f , ir_f], dim=1)


        out = self.decoder0(out)
        out = self.decoder1(out)
        out = self.decoder2(out)
        fus = self.decoder3(out)

        return fus

    def forward(self, ir, vi):
        max = torch.max(ir,vi)
        max = self.suppleConv0(max)
        # max = self.suppleConv1(max)

        #
        out = self.encoder0(ir)
        out = self.encoder1(out)
        ir_f = self.encoder2(out)

        # ir_f_p = ir_f * self.PA(ir_f)
        # ir_f = torch.cat([ir_f, ir_f_p], dim=1)
        # ir_f = self.con(ir_f)
        # ir_f_grad = self.sobel(ir_f)
        # ir_f = ir_f + ir_f_grad
        # ir_f = self.sobelConv(ir_f)

        out = self.encoder0(vi)
        out = self.encoder1(out)
        vi_f = self.encoder2(out)

        # vi_f_grad = self.sobel(vi_f)
        # vi_f = vi_f + vi_f_grad
        # vi_f = self.sobelConv(vi_f)

        # out = torch.cat([vi_f+vi_res , ir_f+ir_res], dim=1)
        out = torch.cat([vi_f , ir_f], dim=1)


        out = self.decoder0(out)
        out = out + max

        # out = self.decoder1(out)
        out = self.decoder1(out)
        fus = self.decoder2(out)

        return fus


class Position_Attention(nn.Module):
    def __init__(self, deploy, use_se):
        super(Position_Attention, self).__init__()
        self.deploy = deploy
        self.use_se = use_se
        self.cur_layer_idx = 1
        self.in_planes = 48
        self.override_groups_map = None or dict()
        self.PA_module =nn.Sequential(OrderedDict([
            ('GAP', nn.AdaptiveAvgPool3d((1, None, None))),
            ('MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),

            ('conv1', nn.Sequential(RepVGGBlock(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se))),
            ('conv1_1', self._make_stage(48, 2, stride=2)),

            ('conv2', nn.Sequential(nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1))),
            ('subpixel_conv1', nn.PixelShuffle(2)),
            ('sigmoid1', nn.Sigmoid()),
            ('conv_k1', nn.Sequential(nn.Conv2d(72, 4, kernel_size=3, stride=1, padding=1))),
            ('subpixel_conv2', nn.PixelShuffle(2)),
            ('sigmoid2', nn.Sigmoid())
        ]))

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    #
    def forward(self, f0):
        f = self.PA_module.GAP(f0)
        f = self.PA_module.MaxPool(f)
        f_conv1 = self.PA_module.conv1(f)
        # f = self.PA_module.AvgPool(f_conv1)
        f = self.PA_module.conv1_1(f_conv1)
        f = self.PA_module.conv2(f)
        f = self.PA_module.subpixel_conv1(f)
        f = self.PA_module.sigmoid1(f)
        f = F.interpolate(f, f_conv1.shape[2:])
        f = torch.cat((f_conv1, f), dim = 1)
        f = self.PA_module.conv_k1(f)
        f = self.PA_module.subpixel_conv2(f)
        f = F.interpolate(f, f0.shape[2:])
        w = self.PA_module.sigmoid2(f)

        return w

class LseRepNet(nn.Module):

    def __init__(self, num_blocks, width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super(LseRepNet, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.use_se = use_se

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.encoder0 = RepVGGBlock(in_channels=1, out_channels=self.in_planes, kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.cur_layer_idx = 1
        self.encoder1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        self.encoder2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=1)

        self.decoder0 = ConvBnLeakyRelu2d(192, 96)
        self.decoder1 = ConvBnLeakyRelu2d(96, 48)
        self.decoder2 = ConvBnLeakyRelu2d(48, 24)
        self.decoder3 = ConvBnTanh2d(24, 1)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.encoder0(x)
        out = self.encoder1(out)
        out = self.encoder2(out)

        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)
        return out

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x
class ConvRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn  = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        return self.nonlinearity(self.conv(x))

class ConvBNRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBNRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        return self.nonlinearity(self.bn(self.conv(x)))

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn  = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
        # return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)

class ConvBnLeakyBNRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyBNRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn  = nn.BatchNorm2d(out_channels)
        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        # return F.leaky_relu(self.conv(x), negative_slope=0.2)
        return self.nonlinearity(self.bn(self.conv(x)))


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return torch.tanh(self.conv(x))/2+0.5
        # return torch.tanh(self.bn(self.conv(x)))/2+0.5
import os
from PIL import Image
import torch
from torchvision import transforms

import os
from PIL import Image
import torch
from torchvision import transforms

# 定义图像处理函数
def preprocess_image(image_path_vis, image_path_ir):
    # 处理可见光图像
    image_vis = Image.open(image_path_vis).convert('L')  
    transform_vis = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    input_vis = transform_vis(image_vis).unsqueeze(0) 

    # 处理红外图像
    image_ir = Image.open(image_path_ir).convert('L') 
    transform_ir = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.5], std=[0.5])  
    ])
    input_ir = transform_ir(image_ir).unsqueeze(0)  

    return input_vis, input_ir


# 定义模型
model = LseRepFusNet(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=False)

# 设置模型为评估模式
model.eval()
