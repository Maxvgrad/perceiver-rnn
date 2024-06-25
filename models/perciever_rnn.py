import torch
import torchvision
from einops.layers.torch import Reduce, Rearrange
from torch import nn


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.linear_stack = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.linear_stack(x)


class UcfClassPredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_classes=11):
        super().__init__()
        self.linear_stack = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.linear_stack(x)


class PerceiverRNN(nn.Module):
    def __init__(self, perceiver, classifier_head, preprocess='None'):
        super().__init__()
        self.perceiver = perceiver
        if preprocess=='dino':
            dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.image_preprocess = dinov2_vits14
        elif preprocess=='cnn':
            self.image_preprocess = torch.nn.Sequential(
                Rearrange('b h w c ->  b c h w'),
                torch.nn.Conv2d(3, 10, 3),
                torch.nn.ReLU(),
                torch.nn.Conv2d(10, 10, 3),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2),
                Rearrange('b c h w -> b h w c'),
            )
        elif preprocess == 'resnet18':
            resnet18 = torchvision.models.resnet18(pretrained=True)
            # https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation
            resnet18.eval()
            resnet18.requires_grad_(requires_grad=False)
            self.image_preprocess = torch.nn.Sequential(
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
            self.image_preprocess = torch.nn.Identity()
        self.classifier_head = classifier_head

    def forward(self, batch, latents=None):
        # batch (B, H, W, C)
        # latents (B, num_latents, latent_dim)
        preprocessed = self.image_preprocess(batch)
        latents = self.perceiver.forward(preprocessed, latents=latents, return_embeddings=True)
        steering = self.classifier_head.forward(latents)
        return steering, latents

