# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn
## Introduction 
This project is a venture into developing a model for autonomous steering for a self-driving task. The challenge for autonomous vehicle is navigating rural roads in Estonia at moderate speed (15-35 km/h), without traffic. In this project we experiment with two deep-learning architectures and report our results. We host our code in the following repository: https://github.com/gorixInc/rally-challenge-24
## Methods
### Dataset and preprocessing
The dataset contains the cropped and antialiased images from the frontal camera of the vehicle. The resolution of the images is 68x264 pixels. We split the dataset with 80% of provided driving runs going to training, and 20% are kept for validation. This means that we train on 41 runs, and validate on 11. The total number of individual frames was 1136580 for the training and 289276 images for the validation set.

![image](https://github.com/gorixInc/rally-challenge-24/assets/73139441/0760b87a-7d1b-4bcc-81a6-9ee098595d08)

For the baseline, we don't augment the dataset in any way. The only preprocessing step that is done is normalization by dividing pixel values of the images by 255. 

The dataset was prepared using using PyTorch native Dataset class extension and ingested to a model using dataloader. Together with a batch of images the steering angle and conditional mask were passed as well.

### Metrics
The model is first evaluated using validation set to yield two metrics
- *MAE* of the steering angle, ie. how different is predicted steering angle from the one used at the frame. The MAE is calculated both total and for straight, right and left marked frames separately.
- *Whiteness* which is a measure of how smooth the steering commands are, measured commonly in degree units per second. The lower values are usually indicating smoother steering.

For the qualitative evaluation of the model we use the VISTA simulation with the two provided traces. The three metrics we use are:
- *Crash score*, i.e the total number of crashes for both test traces.
- *Whiteness*.
- *Effective whiteness*. While the exact definition of this is missing, we assume that this type of whiteness metrics normalized in some way to account for the driving conditions or the specific characteristics of the road (road with lots of sharp turns vs mostly straight road for example).

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

Considering the very low crash score of the trained models, it appears we need to tweak our evaluation procedure.

The programmatic model evaluation of a PilotNet trained on 2 epochs gave next results:
```
{'mae': 9.462068908577107, 'rmse': 29.18889959728998, 'bias': 1.67766252936753, 'max': 906.773631384108, 'whiteness': 183.16432, 'expert_whiteness': 29.571376893873957, 'left_mae': 64.44267043956717, 'straight_mae': 6.996994347305809, 'right_mae': 53.20629443278932}
```
Training one epoch took 4 hours on average. From the results we can see that calculated whiteness can significantly differ from the values given by VISTA simulation, so both methods should be used and results combined together, to get the best idea of model's capabilities. The PilotNet was then trained 2 epochs more to look at the progression of the training and validation loss values.

![image](https://github.com/gorixInc/rally-challenge-24/assets/73139441/80dced38-499c-4b43-9d08-0bdc5b2a0601)

## Conclusion

In the initial part of our project, we detailed our steps and findings in recreating a model for autonomous steering on rural roads in Estonia. We established the dataset, baseline model, and evaluation metrics, and conducted preliminary experiments.

Our baseline model, PilotNet, showed promising results with a crash score of 1 after 2 epochs of training, demonstrating significant improvement over an untrained model. Metrics such as MAE, whiteness, and effective whiteness provided insights into performance and areas for enhancement. Next steps include experimenting with additional architectures like the Perceiver model, refining evaluation procedures, and incorporating data augmentation techniques. 
