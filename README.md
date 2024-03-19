# CAROL

This repository contains main implementation for CAROL: [Certifiably Robust Reinforcement Learning through Model-Based Abstract Interpretation](https://arxiv.org/abs/2301.11374). (SaTML'24)

## Training Requirements
The main code is tested with Python 3.8.12. The required packages are listed in `requirements.txt` and can be intalled with 

`pip install -r requirements.txt`

Additionally, MuJoCo 2.1 is required. 

To run the code, you need add this project's path in the PYTHONPATH with 

`export PYTHONPATH=$PYTHONPATH:/path/to/carol`

## Running
To run the training for experiments in the paper,

`python run.py overrides=hopper`

`python run.py overrides=walker2d`

`python run.py overrides=halfcheetah`

`python run.py overrides=ant`

## Test
We attach an example in `/example_models` with the policy and models for Hopper. `overall` indicates the example policy and the model trained together with this policy. `seperate` contains examples of the separately trained models used for the evaluation in Figure 3. 

For the provability part, a new environment is needed and the packages can be intalled with

`pip install -r requirements_proof.txt`

The main change is for the PyTorch version.

To run provability experiments for this example (with the proof packages):

`cd evaluation`

`python test_provability.py evaluation=provability overrides=hopper`

To run empirical attack on this example:

`cd evaluation`

`python test_attack.py evaluation=attack overrides=hopper`
