# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn
## Introduction 
This project is a venture into developing a model for autonomous steering for a self-driving task. The challenge for autonomous vehicle is navigating rural roads in Estonia at moderate speed (15-35 km/h), without traffic. In this project we experiment with two deep-learning architectures and report our results. We host our code in the following repository: https://github.com/gorixInc/rally-challenge-24
## Methods
### Dataset and preprocessing
The dataset contains the cropped and antialiased images from the frontal camera of the vehicle. The resolution of the images is 68x264 pixels. We split the dataset with 80% of provided driving runs going to training, and 20% are kept for validation. This means that we train on 41 runs, and validate on 11. The total number of individual frames was 1136580 for the training and 289276 images for the validation set.

![image](https://github.com/gorixInc/rally-challenge-24/assets/73139441/0760b87a-7d1b-4bcc-81a6-9ee098595d08)


For the baseline, we don't augment the dataset in any way. The only preprocessing step that is done is normalization by dividing pixel values of the images by 255. 

The dataset was prepared using PyTorch native Dataset class extension and ingested to a model using dataloader. Together with a batch of images the steering angle and conditional mask were passed as well.


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
For the baseline model we're using the PilotNet implementaiton introduced in ["End to End Learning for Self-Driving Cars"][2]. We keep the architecutre uchanged, and use the following layers:
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

#### Perceiver
We adapted the Perceiver model, introduced in the article: Perceiver: General Perception with Iterative Attention (https://arxiv.org/abs/2103.03206). We based our model code on Perceiver implementation by Phil Wang, which can be found here: https://github.com/lucidrains/perceiver-pytorch.  

In our adaptation, we used a single layer of the model as and RNN cell for our time-series task. At each timestep, the image from the frontal camera of the vehicle is passed to a CNN to extract is passed features and reduce dimensionality. The resultant feature map is passed to the Perceiver along with the latent array from ${t-1}$ timestep. We use the latent array from the $t$ timestep to predict the steering angle by passing it to an MLP layer. The latent array is then passed to the next timestep. The model architecutre can be seen here: 

![image](https://github.com/gorixInc/rally-challenge-24/assets/56884921/9c488065-0673-4b3b-bf0d-3a9ef4c08683)

The CNN consists of 2 convolutions with ReLu activation, followed by a max pool layer. The MLP has two linear layers with ReLU activation and the final layer predicting the steering angle. For the Perceiver we used the following parameters:           
 - num_freq_bands = 6      
 - max_freq = 10              
 - depth = 1                  
 - num_latents = 256           
 - latent_dim = 512/64    
 - cross_heads = 1             
 - latent_heads = 4          
 - cross_dim_head = 64
 - latent_dim_head = 64       
 - num_classes = 1             
 - attn_dropout = 0/0.4
 - ff_dropout = 0/0.4
 - fourier_encode_data = True  
 - self_per_cross_attn = 2  (number of self attention blocks)

### Data Loader
(GORDEI & FILLIP)
## Results
### PilotNet

We performed hyperparameter optimization in two stages. In the first stage, we ran optimization on a small subsample of the dataset to optimize hyperparameters such as learning rate, weight decay, batch size, and image augmentation (see results in [Appendix A](#Appendix-A)). In the second stage, we reduced the hyperparameter ranges and ran optimization on the full dataset (see results in [Appendix B](#Appendix-B)).

We then selected two best-performing models on the evaluation dataset, with and without data augmentation. 
The models were subsequently evaluated by running the VISTA evaluation on the official rally competition's test dataset.

#### Data augmentation
Image augmentations such as AddShadow, AddSnowdrops, AddRainStreaks, Gaussian Blur, Random Sharpness Adjustment, and Color Jitter were added to try and train a robust end-to-end driving models. These transformations simulated a wide array of real-world visual conditions including variable lighting, weather effects, and optical variations, which are commonly encountered during driving.

- Weather Simulations (AddShadow, AddSnowdrops, AddRainStreaks): These augmentations mimic different weather conditions like shadows from overhead objects, snowfall, and rain streaks on the lens, helping the model to process and operate under diverse environmental challenges.

- Optical Effects (Gaussian Blur, Random Sharpness Adjustment): These ensure the model can function reliably despite variations in image clarity due to camera focus issues or external factors affecting visibility, such as fog or motion blur.

- Color Variations (Color Jitter): Adjusts image brightness, contrast, and saturation to train the model to recognize important navigational elements under various lighting conditions, essential for tasks like traffic light detection and interpreting road signs.
![img_augments_preview](https://github.com/gorixInc/rally-challenge-24/assets/81022307/8a65bf91-77ad-42a4-92dd-3e7ce4210cb7)

A PilotNet model was trained on the augmented images for 7 epochs. The model was then evaluated by running the VISTA evaluation on the official rally competition's test dataset.

### Perceiver results
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/8eaaa01e-3978-43ec-ba90-2be7e4c30f86" alt="drawing" style="width:400px; margin-right: 10px;"/>
    <img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/6bbda71a-030e-4f2c-9c38-1c04fae7eb26" alt="drawing" style="width:400px;"/>
</div>
<div style="display: flex; justify-content: center; margin-top: 10px;">
    <img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/0c4e2874-0115-4306-bb1c-644acd6c9e13" alt="drawing" style="width:400px;"/>
</div>

### Final results 

Here we have our results compared to our initial baseline model and last year's competition winners ([rally-estonia-challenge-2023-results][1]).

|                           | crash score | avg whiteness | avg eff. whiteness |
|---------------------------|-------------|---------------|--------------------|
| pilotnet-7ep-aug          | 171         | 49.71         | 3.21               |
| pilotnet-without-aug      | 202         | 57.13         | 3.41               |
| baseline-pilotnet-2ep     | 240         | 56.96         | 3.13               |
| Anything_3 (2023 winners) | 167         | -             | 2.718              |


## Conclusion
(GORDEI)


In the initial part of our project, we detailed our steps and findings in recreating a model for autonomous steering on rural roads in Estonia. We established the dataset, baseline model, and evaluation metrics, and conducted preliminary experiments.

Our baseline model, PilotNet, showed promising results with a crash score of 1 after 2 epochs of training, demonstrating significant improvement over an untrained model. Metrics such as MAE, whiteness, and effective whiteness provided insights into performance and areas for enhancement. Next steps include experimenting with additional architectures like the Perceiver model, refining evaluation procedures, and incorporating data augmentation techniques. 

*Project contributions here*

## Appendix-A

![pilotnet-tune-hyperparameter-dataset-short.svg](assets%2Fimages%2Fpilotnet-tune-hyperparameter-dataset-short.svg)

## Appendix-B

| Parameter     | pilotnet-without-aug | pilotnet-7ep-aug |
|---------------|----------------------|------------------|
| augment       | 0                    | 1                |
| epochs        | 10                   | 7                |
| batch_size    | 256                  | 512              |
| learning_rate | 0.000712             | 0.001            |
| weight_decay  | 0.026266             | 0.01             |

![pilotnet-tune-hyperparameter-dataset-full.svg](assets%2Fimages%2Fpilotnet-tune-hyperparameter-dataset-full.svg)

[1]: https://adl.cs.ut.ee/blog/rally-estonia-challenge-2023-results
[2]: https://arxiv.org/abs/1604.07316
