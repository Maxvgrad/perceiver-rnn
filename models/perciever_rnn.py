from torch import nn
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
    def __init__(self, perciever, classifier_head):
        super().__init__()
        self.perciever = perciever
        self.classifier_head = classifier_head

    def forward(self, batch, latents=None):
        # batch (B, H, W, C)
        # latents (B, num_latents, latent_dim)
        latents = self.perciever.forward(batch, latents=latents, return_embeddings=True)
        steering = self.classifier_head.forward(latents)
        return steering, latents

