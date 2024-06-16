# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn

**Project supervisors** Ardi Tampuu, Tambet Matiisen
## Introduction 
This project is a venture into developing a model for autonomous steering for a self-driving task. The challenge for autonomous vehicle is navigating rural roads in Estonia at moderate speed (15-35 km/h), without traffic. In this project we apply two deep-learning architectures to this task - PilotNet and Perceiver, and report our results. We host the code in our repository: https://github.com/gorixInc/rally-challenge-24
## Methods
### Dataset and preprocessing
The dataset contains the cropped and antialiased images from the frontal camera of the vehicle. The resolution of the images is 68x264 pixels. We split the dataset with 80% of provided driving runs going to training, and 20% are kept for validation. This means that we train on 41 runs, and validate on 11. The total number of individual frames was 1.4 million. The only preprocessing step that is done is normalization by dividing pixel values of the images by 255. Below is an example of an image from the dataset.

![image](https://github.com/gorixInc/rally-challenge-24/assets/73139441/0760b87a-7d1b-4bcc-81a6-9ee098595d08)

<!-- 
For the baseline, we don't augment the dataset in any way.  

The dataset was prepared using PyTorch native Dataset class extension and ingested to a model using dataloader. Together with a batch of images the steering angle and conditional mask were passed as well. -->


### Metrics
The model is first evaluated using validation set to yield two metrics
- *MAE* of the steering angle, ie. how different is predicted steering angle from the one used at the frame. The MAE is calculated both total and for straight, right and left marked frames separately.
- *Whiteness*, which is a measure of how smooth the steering commands are, measured commonly in degree units per second. The lower values are usually indicating smoother steering.

For the qualitative evaluation of the model we use the VISTA simulation with the two provided traces. The three metrics we use are:
- *Crash score*, i.e the total number of crashes for both test traces.
- *Whiteness*.
- *Effective whiteness*. While the exact definition of this is missing, we assume that this type of whiteness metrics normalized in some way to account for the driving conditions or the specific characteristics of the road (road with lots of sharp turns vs mostly straight road for example).

### Models
#### Baseline 
For the baseline model we used the PilotNet implementaiton introduced in ["End to End Learning for Self-Driving Cars"][2]. We keep the architecutre uchanged, and use the following layers:
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
We adapted the Perceiver model, introduced in the article: ["Perceiver: General Perception with Iterative Attention"][3], basing our implementation on the [perceiver-pytorch implementation by Phil Wang][4].

In our adaptation, a single layer of the model is used as an RNN cell for our time-series task. At each timestep, the image from the frontal camera of the vehicle is passed to a CNN to extract features and reduce dimensionality. The resultant feature maps are passed to the Perceiver along with the latent array from ${t-1}$ timestep. We use the latent array from the $t$ timestep to predict the steering angle by passing it to an MLP layer. The latent array is then passed to the next timestep. The model architecutre can be seen here: 

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

The data loader implementation for both PilotNet and Perceiver was based on the `NvidiaDataset()` class presented in the [e2e-rally-estonia repository][5]. During the loader initialization, the desired transformations and color space can be chosen. For each driven path, a pandas DataFrame is created, which contains the file paths to each image from the camera and metadata, including steering angle, vehicle speed, turn signal data, etc. The DataFrames for each separate driven path are then concatenated into a single DataFrame, containing image paths and metadata for all frames across all paths. For PilotNet, the dataloader extracts a batch of images and corresponding steering angles.


<!-- The mean absolute error (MAE) was shown (https://www.mdpi.com/1424-8220/23/5/2845) to be a more suitable loss function for this task than mean squared error (MSE). Therefore, MAE was primarily used. The Adam optimizer with weight decay was employed, and to prevent overfitting, early stopping was implemented with a patience of 10 epochs without a decrease in validation loss. -->



The RNN version of the dataset and data loader was based mainly on the PilotNet version, but with specific adaptations for the RNN architecture. The unshuffled frames in the form of a DataFrame are initially divided into a number of sequences of certain length with stride stride between the sequences. Both length and stride are configurable parameters. During each iteration of the data loader, a batch of sequences is extracted, maintaining the chronological order of the frames. 

<!-- Each frame is forward-passed to the model separately, together with a latent array. The loss is then calculated for each time step in the sequence, along with a set of metrics at the end of each epoch. -->

We also optimized the the dataloaders by first converting all images from a given driven path into a PyTorch tensor and saving them to disk. Durning training, the tensors are loaded and cached accordingly in the DataSet instance. This allows for much faster training when using unshuffled DataLoader thanks to a reduction in storage IOPS, compared to loading each image from disk individually.


## Results
### PilotNet

We performed hyperparameter optimization in two stages. In the first stage, we ran optimization on a small subsample of the dataset to optimize hyperparameters such as learning rate, weight decay, batch size, and image augmentation (see results in [Appendix A](#Appendix-A)). In the second stage, we reduced the hyperparameter ranges and ran optimization on the full dataset (see results in [Appendix B](#Appendix-B)).

We then selected two best-performing models on the evaluation dataset, with and without data augmentation. However, we did not obtain VISTA evaluation results for the former due to time constraints. 

#### Data augmentation
Image augmentations such as AddShadow, AddSnowdrops, AddRainStreaks, Gaussian Blur, Random Sharpness Adjustment, and Color Jitter were added to try and train a robust end-to-end driving models. These transformations simulated a wide array of real-world visual conditions including variable lighting, weather effects, and optical variations, which are commonly encountered during driving.

- Weather Simulations (AddShadow, AddSnowdrops, AddRainStreaks): These augmentations mimic different weather conditions like shadows from overhead objects, snowfall, and rain streaks on the lens, helping the model to process and operate under diverse environmental challenges.

- Optical Effects (Gaussian Blur, Random Sharpness Adjustment): These ensure the model can function reliably despite variations in image clarity due to camera focus issues or external factors affecting visibility, such as fog or motion blur.

- Color Variations (Color Jitter): Adjusts image brightness, contrast, and saturation to train the model to recognize important navigational elements under various lighting conditions, essential for tasks like traffic light detection and interpreting road signs.
![img_augments_preview](https://github.com/gorixInc/rally-challenge-24/assets/81022307/8a65bf91-77ad-42a4-92dd-3e7ce4210cb7)

We trained a PilotNet model on the augmented images for 7 epochs.

### Perceiver results
For our experiments with the Perceiver we did not use data augmentation and trained on only 4 paths from the dataset as to iterate on the model faster. The models were trained with sequences of images of length 128. For all our tests with different parameters, we observed models very quickly converging to local minima with very poor performance. For fixed batch size and sequence length, all models converged to approximately the same high loss value, high prediction RMSE, and in most cases approached a 0 whiteness score, meaning the models likely predicted a constant steering angle.
In the figure below you can observe the described effects:

<div align='left' class="image-row">
<img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/06f06ce5-cb91-4a57-90e8-ce18ce4c70ef" alt="drawing" style="width:400px;"/>
<img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/300a583a-c4a8-4922-b85b-185c4a3e79e6" alt="drawing" style="width:400px;"/>

<img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/29754aee-e5cb-42d1-b689-3b0bca37e8d3" alt="drawing" style="width:400px;"/>
<img src="https://github.com/gorixInc/rally-challenge-24/assets/56884921/4b3128e3-48e2-46a5-a597-bb81dabd51f4" alt="drawing" style="width:400px;"/>
</div>

It remains unclear exactly what caused these issues with the architecture. Some possible explanations include incorrect Perceiver configuration or lack of more sophisticated regularization methods for our purpose.


### Final PilotNet results 

Here we present the results two best models we obtained along with our initial baseline model and last year's competition winner ([rally-estonia-challenge-2023-results][1]). 

|                           | crash score | avg whiteness | avg eff. whiteness |
|---------------------------|-------------|---------------|--------------------|
| pilotnet-7ep-aug (steering)          | 171         | 49.71         | 3.21               |
| tuned-pilotnet-without-aug (steering)    | 202         | 57.13         | 3.41               |
| baseline-pilotnet-2ep (steering)     | 240         | 56.96         | 3.13               |
| Anything_3 (conditional, 2023 winners) | 167         | -             | 2.718              |

The parameters for the models are in Appendix B. Our models were tested by running the VISTA evaluation on the official rally competition's test dataset with `road_width = 2`.
## Conclusion

In this project, we experimented with two model architectures: PilotNet and Perceiver.

Our results demonstrate that data augmentation strategies improved the performance of the PilotNet model. PilotNet model achieved a slightly poorer crash score than the Rally Estonia competition winner of 2023, but was better than the model that took second place.

We adapted the Perceiver architecture to work with frames sequences rather. Despite exploring various configurations and training on a limited dataset for faster iteration, the Perceiver models consistently converged to suboptimal solutions. Further work is required to address the limitations observed with our adapted Perceiver architecture.

### Contributions
- Maksim Ploter: implementing training infrastructure, wandb set-up and implementation, Pilotnet tuning and evaluation
- Gordei Pribõtkin: implementing models, implementing RNN dataloader, Perceiver evaluation, VISTA set-up, dataloader optimization
- Filips Petuhovs: implementing dataloaders, implementing trainers, initial experiments
- Rain Eichhorn: VISTA set-up, implementing data preprocessing, PilotNet evaluation

All project members contributed equally to the blogpost.

## Appendix-A

![pilotnet-tune-hyperparameter-dataset-short.svg](https://raw.githubusercontent.com/gorixInc/rally-challenge-24/4fd53b46dcfbac89f5e293d065d9da04ce07cbe6/assets/images/pilotnet-tune-hyperparameter-dataset-short.svg)

## Appendix-B

| Parameter     | tuned-pilotnet-without-aug | pilotnet-7ep-aug |
|---------------|----------------------|------------------|
| augment       | 0                    | 1                |
| epochs        | 10                   | 7                |
| batch_size    | 256                  | 512              |
| learning_rate | 0.000712             | 0.001            |
| weight_decay  | 0.026266             | 0.01             |

![pilotnet-tune-hyperparameter-dataset-full.svg](https://raw.githubusercontent.com/gorixInc/rally-challenge-24/4fd53b46dcfbac89f5e293d065d9da04ce07cbe6/assets/images/pilotnet-tune-hyperparameter-dataset-full.svg)

[1]: https://adl.cs.ut.ee/blog/rally-estonia-challenge-2023-results
[2]: https://arxiv.org/abs/1604.07316
[3]: https://arxiv.org/abs/1604.07316
[4]: https://github.com/lucidrains/perceiver-pytorch 
[5]: https://github.com/UT-ADL/e2e-rally-estonia
