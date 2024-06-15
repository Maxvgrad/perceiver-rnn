from torch import nn
import torch
from einops.layers.torch import Reduce

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

class PerceiverRNN(nn.Module):
    def __init__(self, perciever, classifier_head, use_dino=False):
        super().__init__()
        self.perciever = perciever
        if use_dino:
            dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.image_preprocess = dinov2_vits14
        else:
            self.image_preprocess = torch.nn.Identity()
        self.classifier_head = classifier_head

    def forward(self, batch, latents=None):
        # batch (B, H, W, C)
        # latents (B, num_latents, latent_dim)
        preprocessed = self.image_preprocess(batch)
        latents = self.perciever.forward(preprocessed, latents=latents, return_embeddings=True)
        steering = self.classifier_head.forward(latents)
        return steering, latents

