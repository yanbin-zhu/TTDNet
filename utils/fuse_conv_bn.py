import torch
import torch.nn as nn

def fuse_conv_bn(conv, bn):

    conv_weight = conv.weight
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)


    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    bn_eps = bn.eps

    factor = bn_weight / torch.sqrt(bn_running_var + bn_eps)


    fused_weight = conv_weight * factor.reshape([conv.out_channels, 1, 1, 1])


    fused_bias = (conv_bias - bn_running_mean) * factor + bn_bias


    conv.weight = nn.Parameter(fused_weight)
    conv.bias = nn.Parameter(fused_bias)

    return conv

def fuse_module(module):
    last_conv = None
    last_conv_name = None


    for name, child in module.named_children():

        if isinstance(child, (nn.BatchNorm2d, nn.SyncBatchNorm)):

            if last_conv is not None:
                fused_conv = fuse_conv_bn(last_conv, child)
                module._modules[last_conv_name] = fused_conv

                module._modules[name] = nn.Identity()
                last_conv = None


        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name


        else:
            fuse_module(child)

    return module