# The Rally Estonia Challenge 2024
**Project members:** Maksim Ploter, Gordei Pribõtkin, Filips Petuhovs, Rain Eichhorn

**Project supervisors:** Ardi Tammpuu, Tambet Matiisen
## Motivation
The goal is to train a neural network that transforms forward-facing camera feed into steering commands. Additionally, the model can be made to be conditional: depending on either directional input or coordinate waypoints yielding different steering commands. We will apply novel model architecture(s) and evaluate their performance. This project will allow us to engage with real-time image processing, agents in machine learning and cutting-edge deep learning architectures. We are also tremendously excited to try out our model on a real car.

## Usage
### Evaluation 
- Put traces into vista_eval/traces
- `bash cd vista_eval`
- `sbatch evaluate_slurm.sh --model ./models/<MODEL_NAME> --traces <TRACE_FOLDER_NAME> --traces-root ./traces/ --save-video`
