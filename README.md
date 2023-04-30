# Probing Emergent World Representations in Attention-Based LLMs: GPT trained to play Othello

## Summary:
In this project, we extend the investigations presented by Kenneth Li et al. in their ICLR2023 Paper [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/pdf/2210.13382.pdf). We first show that trained linear probes can accurately map the activation vectors in a GPT model pre-trained to play legal moves in the game Othello. This is in contrast to the utilization of the non-linear MLP as probes. We also show that changes to the "world model", visualized by the linear probes, are causal with respect to the GPT next-move predictions, demonstrated for complex board changes. Lastly, we train several GPT models of different sizes (number of attention blocks and number of heads) and show that the world representation only appears for the deep models. 

## Inside the Repository:
- Team scripts are in folders dev_code2/ 
- Original scripts from the official paper repository are included in folder EWOthello/KLiScripts/ for reference. 

### Install Locally
After cloning the repository, create a new conda environment (or equivalent). Then cd to the folder and install the codebase EWOthello as a python package via,
```
python setup.py develop
```
You can then install additional dependencies via
```
pip install -r requirements.txt
```
You should then be able to run all scripts. It is also possible to run the code in this repository (unverified) on google collab by adding the following lines:
'''
!git lfs install
!git clone https://github.com/DeanHazineh/Emergent-World-Representations-Othello
%cd /content/Emergent-World-Representations-Othello
!python setup.py develop
'''
