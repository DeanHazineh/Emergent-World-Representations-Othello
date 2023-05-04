# Probing Emergent World Representations in Transformer Networks: Sequential Models Trained to Play Othello

## Summary:
In this project, we extend the investigations presented by Kenneth Li et al. in their ICLR 2023 Paper [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/pdf/2210.13382.pdf). We contribute the following new insights: 

- We first show that trained linear probes can accurately map the activation vectors of a GPT-model, pre-trained to play legal moves in the game Othello, to the current state of the othello board. This is in contrast to the utilization of the non-linear MLP as probes. 
- We also show that changes to the "world model", visualized by the linear probes, are causal with respect to the model's next-move predictions. This is demonstrated for complex board changes in addition to single tile flips. By modifying the intervention scheme, we provide new insights that the causality of the world representation depends on the layer.
- We train several sequential models of different sizes (number of attention blocks and number of heads) and analyze the effectiveness of interventions based on the world representation. Our findings suggest that although encoded, the world representation is not used in the model's predictions for the final layers of the model. Our results suggest semantic information is formed and utilized mid-way through deep, transformer-based models.
- We also apply transformer circuit theory and find regularities in the attention heads, which can be categorized into "my-turn" vs "your-turn" heads. Furthermore, we observe that the attention heads in the deep layers (layer 7 and 8 of the 8 layer model for example) have non-zero values mostly contained on the diagonal or in the first column meaning that these attention heads contribute little to the next move prediction. This is consistent with the probe causality findings. 

## Inside the Repository:
The scripts to produce all figures, data, and trained models are included in folders dev_code2/. Original scripts from the official paper repository are included in folder EWOthello/KLiScripts/ for reference. 

An interactive colab document to play with the attention heads can be found https://colab.research.google.com/drive/1OZjK_axPx4jiGT4-Kpy2uzCFpB6YWAD-?usp=sharing

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
