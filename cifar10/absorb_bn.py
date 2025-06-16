import torch
import torch.nn as nn
from quant_dorefa import *
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, model_parameters
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler
from collections import OrderedDict
from model import *

def fuse_conv_and_bn(conv, bn):
    with torch.no_grad():
        fusedconv = Conv2d_Q(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        beta = bn.weight
        gamma = bn.bias
        if conv.bias is not None:
            b = conv.bias
        else:
            b = mean.new_zeros(mean.shape)
        w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean) / var_sqrt * beta + gamma

        fusedconv.bias = nn.Parameter(b)
        fusedconv.weight = nn.Parameter(w)
    return fusedconv

def fuse_linear_and_bn(linear, bn):
    with torch.no_grad():
        W = linear.weight.data
        b = linear.bias.data if linear.bias is not None else torch.zeros(W.size(0))

        gamma = bn.weight.data
        beta = bn.bias.data
        mu = bn.running_mean
        sigma = torch.sqrt(bn.running_var + bn.eps)

        W_fused = gamma.view(-1, 1) * (W / sigma.view(-1, 1))
        b_fused = gamma * (b - mu) / sigma + beta

        fused_linear = Linear_Q(linear.in_features, linear.out_features)
        fused_linear.weight.data = W_fused
        fused_linear.bias.data = b_fused
    return fused_linear

def main():
    model = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=12, mlp_ratios=4,
        in_channels=3, num_classes=10, qkv_bias=False,
        depths=4, sr_ratios=1,
        T=4)

    checkpoint_path = 'model_best.pth.tar'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    for m in model.modules():
        debug_dot = m
        if isinstance(m, SPS):
            fused_conv = fuse_conv_and_bn(m.proj_conv, m.proj_bn)
            m.proj_conv = fused_conv
            # m.proj_bn = nn.Identity()
            fused_conv = fuse_conv_and_bn(m.proj_conv1, m.proj_bn1)
            m.proj_conv1 = fused_conv
            # m.proj_bn1 = nn.Identity()
            fused_conv = fuse_conv_and_bn(m.proj_conv2, m.proj_bn2)
            m.proj_conv2 = fused_conv
            # m.proj_bn2 = nn.Identity()
            fused_conv = fuse_conv_and_bn(m.proj_conv3, m.proj_bn3)
            m.proj_conv3 = fused_conv
            # m.proj_bn3 = nn.Identity()
            fused_conv = fuse_conv_and_bn(m.rpe_conv, m.rpe_bn)
            m.rpe_conv = fused_conv
            # m.rpe_bn = nn.Identity()

        elif isinstance(m, Block):
            for mm in m.modules():
                if isinstance(mm, SSA):
                    fused_linear = fuse_linear_and_bn(mm.q_linear, mm.q_bn)  # q
                    mm.q_linear = fused_linear
                    # mm.q_bn = nn.Identity()

                    fused_linear = fuse_linear_and_bn(mm.k_linear, mm.k_bn)  # k
                    mm.k_linear = fused_linear
                    # mm.k_bn = nn.Identity()

                    fused_linear = fuse_linear_and_bn(mm.v_linear, mm.v_bn)  # v
                    mm.v_linear = fused_linear
                    # mm.v_bn = nn.Identity()

                    fused_linear = fuse_linear_and_bn(mm.proj_linear, mm.proj_bn)  # proj
                    mm.proj_linear = fused_linear
                    # mm.proj_bn = nn.Identity()

                elif isinstance(mm, MLP):
                    fused_linear = fuse_linear_and_bn(mm.fc1_linear, mm.fc1_bn)  # fc1
                    mm.fc1_linear = fused_linear
                    # mm.fc1_bn = nn.Identity()

                    fused_linear = fuse_linear_and_bn(mm.fc2_linear, mm.fc2_bn)  # fc1
                    mm.fc2_linear = fused_linear
                    # mm.fc2_bn = nn.Identity()
    print(1)
    checkpoint['state_dict'] = model.state_dict()
    checkpoint['epoch'] = 0
    checkpoint.pop('optimizer')
    torch.save(checkpoint, 'model_bn_absort.pth.tar')

if __name__ == '__main__':
    main()
    # checkpoint_path = 'model_bn_absort.pth.tar'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    #
    # checkpoint_path1 = 'model_best.pth.tar'
    # realpth = torch.load(checkpoint_path1, map_location='cpu')
    # print(1)