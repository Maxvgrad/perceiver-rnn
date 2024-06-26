# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn

**Project supervisors:** Ardi Tammpuu, Tambet Matiisen
## Motivation
The goal is to train a neural network that transforms forward-facing camera feed into steering commands. Additionally, the model can be made to be conditional: depending on either directional input or coordinate waypoints yielding different steering commands. We will apply novel model architecture(s) and evaluate their performance. This project will allow us to engage with real-time image processing, agents in machine learning and cutting-edge deep learning architectures. We are also tremendously excited to try out our model on a real car.

## Usage

### Evaluation 

```
conda env create -f environment.yml
conda activate rally_challenge_env
```

### Pulling vista submodule
First time:
```
git submodule update --init --recursive
```

After:
```
git submodule update --recursive
```
### Evaluation 
1. Set up conda env:
     ```
    cd vista_eval
    conda create -n vista python=3.8
    conda activate vista
    pip install -r requirements.txt
    pip install git+https://github.com/UT-ADL/vista.git
    ```

1. Download traces
    ```
    cd vista_eval
    sh download_traces.sh
    ```

1. Run evaluation slurm:
    ```
    cd vista_eval
    conda activate vista
    ```
    ```
    mkdir vista_slurm
    sbatch evaluate_slurm.sh --model ./models/<MODEL_NAME> --traces forward_trace backward_trace --traces-root ./traces/ --save-video
    ```
### Setting Up Training Environment
The training can be done in [HPC Centre’s JupyterHub](https://docs.hpc.ut.ee/course/lab5/#jupyter). For that, the initial preparations must be done once, after which the only preliminary step required is choosing the right Jupyter kernel.
First, login to HPC from the terminal using
```
ssh <username>@rocket.hpc.ut.ee
```
Load the required modules.
```
module load python/3.8.6
module load cuda/11.7.0
module load lzma/4.32.7
module load xz/5.2.5
```
Create a new Python virtual environment. This will store all our dependencies needed to run IPython Kernel.
```
python3 -m venv <name-of-the-environment>
source venv/bin/activate
```
Upgrade pip to the latest version. Then install ipykernel dependency and add it to the JupyterLab environment.
```
pip3 install --upgrade pip
pip3 install ipykernel
python3 -m ipykernel install --user --name=<name-of-the-kernel>
```
Install all the remaining dependencies to your virtual environment using `requirements.txt`.
```
pip3 install -r requirements.txt
```
Now we are done in the terminal. Move to the JupyterLab (or start the new one choosing resource option you need and then cloning into this repo through the GUI) and inside of a notebook open the kernel selection menu on the top-right corner.

![image](https://github.com/gorixInc/rally-challenge-24/assets/73139441/9a8dcb0b-4b07-449f-95bb-37c2ca741fc0)

There you should see your created kernel under **Start Preferred Kernel** section. Select it.

Next time if you want to run any tests in the notebook, you should only check that your kernel is selected. You can add freely any other dependencies if you need by running `pip3 install` while being inside of the virtual environment. These dependencies will be automatically synchronized with the ipykernel. Also don't forget to add any new dependencies you want to include permanently to the **requirements.txt** file.

**[Optional]** Initialize weights and biases library `wandb`. It is highly recommended as it ease up tracking your experiments.
    - Create an account at [wandb.ai](http://wandb.ai) then login
    
    ```bash
    wandb login
    ```
    
    - More information about `wandb`  [here](https://docs.wandb.ai/guide)
  
## Converting original dataset for training
- From project directory, run: `sbatch convert_to_tensor_dataset.sh.dev`
- This will take a couple of hours. A folder will be created in project directory called `tensor_dataset` and will be filled with subfloders. 
- Check that the sbatch is running, as it asks for 64G memory.
