# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn

**Project supervisors:** Ardi Tammpuu, Tambet Matiisen
## Motivation
The goal is to train a neural network that transforms forward-facing camera feed into steering commands. Additionally, the model can be made to be conditional: depending on either directional input or coordinate waypoints yielding different steering commands. We will apply novel model architecture(s) and evaluate their performance. This project will allow us to engage with real-time image processing, agents in machine learning and cutting-edge deep learning architectures. We are also tremendously excited to try out our model on a real car.

## Usage
### Pulling vista submodule
First time:
- `git submodule update --init --recursive`

After:
- `git submodule update --recursive`
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
   - Forward: `sbatch evaluate_slurm.sh --model ./models/<MODEL_NAME> --traces forward_trace --traces-root ./traces/ --save-video`
   - Backward: `sbatch evaluate_slurm.sh --model ./models/<MODEL_NAME> --traces backward_trace --traces-root ./traces/ --save-video`
