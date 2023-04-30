# Probing Emergent World Representations in Attention-Based LLMs: GPT trained to play Othello

## Summary:
In this project, we extend the investigation presented by Kenneth Li et al. in their ICLR2023 Paper "EMERGENT WORLD REPRESENTATIONS: EXPLORING A
SEQUENCE MODEL TRAINED ON A SYNTHETIC TASK" (https://arxiv.org/pdf/2210.13382.pdf). 

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
