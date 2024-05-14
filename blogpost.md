# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn
## Introduction 
This project is a venture into developing a model for autonomous steering for a self-driving task. The challenge for autonomous vehicle is navigating rural roads in Estonia at moderate speed (15-35 km/h), without traffic. In this project we experiment with *two-three?* deep-learning architectures and report our results.  

## Methods
### Dataset and preprocessing
The dataset contains the cropped and antialiased images from the frontal camera of the vehicle. The resolution of the images is 68x264 pixels. We split the dataset with 80% of provided driving runs going to training, and 20% are kept for validation. This means that we train on 41 runs, and validate on 11. 

For the baseline, we don't augment the dataset in any way. The only preprocessing step that is done is normalization by dividing pixel values of the images by 255. 

### Metrics
For the evaluation of the model we use the VISTA simulation with the two provided traces. The two metrics we use are:
- *Crash score*, i.e the total number of crashes for both test traces.
- *Whiteness*, which is a measure of how smooth the steering commands are, with lower values indicating smoother steering.
- *Effective whiteness*, ...

### Models
#### Baseline 
For the baseline model we're using the PilotNet implementaiton introduced in https://arxiv.org/pdf/1604.07316. We keep the architecutre uchanged, and use the following layers:
```
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
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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
```

### Perceiver
TODO

## Results
### Preliminary experiments baseline model
We trained the Baseline PilotNet model for 2 epochs and ran the VISTA evaluation. To put our results into perspective, we also ran the same evaluation on untrained baseline PilotNet, and also the two conditional models provided with VISTA evaluation repository - steering-2 and steering-overfit. Below are the results

|                        | crash score | avg whiteness | avg eff. whiteness |
|------------------------|-------------|---------------|--------------------|
| baseline-pilotnet-2ep  | 1           | 37.95         | 0.96               |
| baseline-pilotnet-untr | 155         | 0.01          | 2.33               |
| steering-2             | 2           | 39.48         | 0.83               |
| steering-overfit       | 2           | 22.70         | 0.74               |

Considering the very low crash score of our models, it appears we need to tweak our evaluation procedure.

## Conclusion