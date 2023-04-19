import os
import pickle
from tqdm import tqdm
import torch

from EWOthello.mingpt.dataset import CharDataset
from EWOthello.data.othello import *
from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing_v2

device = torch.cuda.current_device()
probe_layer = 6
savepath = "EWOthello/data/othello_synthetic_probeTrainingSet/"


def gen_and_save_Probe_Data(model_probe, num_pairs=1e6):
    # Freeze the probe model since we are only using it to grab activations
    for param in model_probe.parameters():
        param.requires_grad = False

    synthetic_dat_path = "EWOthello/data/othello_synthetic/"
    listfiles = os.listdir(synthetic_dat_path)

    # Loop over files to avoid crashing memory
    properties_modifier_matrix = np.ones((59, 64))
    for i in range(59):
        if i % 2 == 1:
            properties_modifier_matrix[i, :] *= -1.0

    for file in listfiles:
        with open(synthetic_dat_path + file, "rb") as handle:
            game_data = pickle.load(handle)
        training_dataset = CharDataset(game_data)

        property_container_v2 = []
        property_container = []
        act_container = []
        for x, _ in tqdm(training_dataset):
            # Convert the game index sequence to board number sequence for use with the othello board class
            tbf = [training_dataset.itos[_] for _ in x.tolist()]
            valid_until = tbf.index(-100) if -100 in tbf else 999

            # Get the board state vectors
            a = OthelloBoardState()
            properties = a.get_gt(tbf[:valid_until], "get_state")
            # property_container.extend(properties)
            properties_v2 = (np.array(properties) - 1.0) * properties_modifier_matrix[:valid_until, :] + 1.0
            property_container_v2.extend(properties_v2.tolist())

            # Get the activation vectors
            act = model_probe(x[None, :].to(device))
            act = torch.stack(act).transpose(0, 1).detach().cpu().tolist()
            act_container.extend(act)
            del a

        data = {"tbf": tbf, "property_container": property_container, "property_container_v2": property_container_v2, "act_container": act_container}
        with open(savepath + file, "wb") as fhandle:
            pickle.dump(data, fhandle)

        break

    return


if __name__ == "__main__":
    # Load the GPT Othello Model
    mconf = GPTConfig(vocab_size=61, block_size=59, n_layer=8, n_head=8, n_embd=512)
    model_probe = GPTforProbing_v2(mconf, probe_layer=probe_layer)

    mode = "synthetic"
    if mode == "random":
        model = GPT(mconf)
        model_probe.apply(model._init_weights)
    else:
        path = (
            "/home/deanhazineh/Research/emergent_world_representation/EWOthello/ckpts/gpt_championship.ckpt"
            if mode == "championship"
            else "/home/deanhazineh/Research/emergent_world_representation/EWOthello/ckpts/gpt_synthetic.ckpt"
        )
        model_probe.load_state_dict(torch.load(path))
    if torch.cuda.is_available():
        model_probe = model_probe.to(device)

    gen_and_save_Probe_Data(model_probe)
