# Probing Emergent World Representations in Attention-Based LLMs: GPT trained to play Othello

## Summary:
In this project, we extend the investigations presented by Kenneth Li et al. in their ICLR2023 Paper [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/pdf/2210.13382.pdf). We contribute the following new insights: 

- We first show that trained linear probes can accurately map the activation vectors of a GPT model, pre-trained to play legal moves in the game Othello, to the current state of the othello board. This is in contrast to the utilization of the non-linear MLP as probes. 
<img src=/images/new_world_representation.png alt="drawing" width="100%"/>

- We also show that changes to the "world model", visualized by the linear probes, are causal with respect to the GPT next-move predictions. This is demonstrated for complex board changes in addition to single tile flips. By modifying the intervention scheme, we demonstrate that in all cases, the causality of the world representation depends on the probe-layer.
<img src=/images/overview.png alt="drawing" width="100%"/>
<img src=/images/Example_intervention.png alt="drawing" width="100%"/>
<img src=/images/new_latent_saliency.png alt="drawing" width="100%"/>

- Lastly, we train several GPT models of different sizes (number of attention blocks and number of heads) and find that the causal world representations provide succesful interventions for deeper models vs shallow. From these results, we hypothesize that although all models achieve nearly 100% accuracy in playing legal othello games, the (unsupervised) learned internal representations differs when increasing the network depth. 
<img src=/images/GPT_models.png alt="drawing" width="100%"/>
<img src=/images/linear_probes.png alt="drawing" width="100%"/>
<img src=/images/causal_intervention_comparison.png alt="drawing" width="100%"/>

## Inside the Repository:
The scripts to produce all figures, data, and trained models are included in folders dev_code2/. Original scripts from the official paper repository are included in folder EWOthello/KLiScripts/ for reference. 

#### Installation Instructions:
After cloning the repository, create a new conda environment (or equivalent). Then cd to the folder and install the codebase EWOthello as a python package via,
```
python setup.py develop
```
You can then install additional dependencies via
```
pip install -r requirements.txt
```
You should then be able to run all scripts. It is also possible to run the code in this repository (unverified) on google collab by adding the following lines:
```
!git lfs install
!git clone https://github.com/DeanHazineh/Emergent-World-Representations-Othello
%cd /content/Emergent-World-Representations-Othello
!python setup.py develop
```
