from torch import nn

class PilotNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear_stack = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(),
            nn.Linear(100, 50), 
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.linear_stack(x)
        return x