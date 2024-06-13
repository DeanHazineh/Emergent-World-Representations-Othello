import torch 

from saes.sae import SAEAnthropic, train_model
from saes.utils import load_pre_trained_gpt
from EWOthello.data.othello import get
from EWOthello.mingpt.dataset import CharDataset
from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing, GPTforProbing_v2
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'


if __name__=="__main__":
    print("Beginning training process. It may take a moment to load the datasets...")
    probe_path = "EWOthello/ckpts/DeanKLi_GPT_Synthetic_8L8H/"
    probe_layer = 6
    GPT_probe=load_pre_trained_gpt(probe_path=probe_path, probe_layer=probe_layer)

    train_dataset = CharDataset(get(ood_num=-1, data_root=None, num_preload=11)) # 11 corresponds to over 1 million games

    test_dataset = CharDataset(get(ood_num=-1, data_root=None, num_preload=1))
    test_set_indices=torch.arange(1000)
    test_1k_dataset = torch.utils.data.Subset(test_dataset, test_set_indices)

    print("\n\n\n")
    print(len(test_dataset))
    print("\n\n\n")

    sae=SAEAnthropic(gpt=GPT_probe, feature_ratio=2, sparsity_coefficient=.1, window_start_trim=4, window_end_trim=4)
    print("SAE initialized, proceeding to train!")

    train_model(sae, train_dataset, test_1k_dataset, report_every_n_steps=500)