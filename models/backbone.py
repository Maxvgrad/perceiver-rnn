import torch
import torchvision
from einops.layers.chainer import Rearrange


def build_image_preprocess_backbone(args):
    if args.perceiver_img_pre_type == 'dino':
        return torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.perceiver_img_pre_type == 'cnn':
        return torch.nn.Sequential(
            Rearrange('b h w c ->  b c h w'),
            torch.nn.Conv2d(3, 10, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(10, 10, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            Rearrange('b c h w -> b h w c'),
        )
    elif args.perceiver_img_pre_type == 'resnet18':
        resnet18 = torchvision.models.resnet18(pretrained=True)
        # https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation
        resnet18.eval()
        resnet18.requires_grad_(requires_grad=False)
        return torch.nn.Sequential(
            Rearrange('b h w c ->  b c h w'),
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool,
            Rearrange('b c h w -> b h w c'),
        )
    else:
        return torch.nn.Identity()
