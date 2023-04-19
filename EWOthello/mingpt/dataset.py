import itertools
import torch
from torch.utils.data import Dataset
import numpy as np

from EWOthello.mingpt.model import GPT, GPTConfig, GPTforProbing
from EWOthello.data.othello import OthelloBoardState


class CharDataset(Dataset):
    def __init__(self, data):
        if hasattr(data, "ood_perc"):
            ood_perc = data.ood_perc
            data.ood_perc = 0  # shut down the randomness
        chars = sorted(list(set(list(itertools.chain.from_iterable(data)))) + [-100])

        data_size, vocab_size = len(data), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(data[_]) for _ in range(len(data))])  # should be 60 in Othello
        print("Dataset created has %d sequences, %d unique words." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = max_len - 1  # for autoregressive training (always train on sequences of fixed max length, aka block)
        self.vocab_size = vocab_size
        if hasattr(data, "ood_perc"):
            data.ood_perc = ood_perc  # turn on the randomness
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) != self.max_len:
            chunk += [
                -100,
            ] * (
                self.max_len - len(chunk)
            )  # -100 can be ignored in CE
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


class ProbingDataset(Dataset):
    def __init__(self, act, y, age):
        assert len(act) == len(y)
        assert len(act) == len(age)
        print(f"{len(act)} pairs loaded...")
        self.act = act
        self.y = y
        self.age = age
        print(np.sum(np.array(y) == 0), np.sum(np.array(y) == 1), np.sum(np.array(y) == 2))

        long_age = []
        for a in age:
            long_age.extend(a)
        long_age = np.array(long_age)
        counts = [np.count_nonzero(long_age == i) for i in range(60)]
        del long_age
        print(counts)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.act[idx], torch.tensor(self.y[idx]).to(torch.long), torch.tensor(self.age[idx]).to(torch.long)


### Dean modification of the probing dataset which enables the new board state representation
class probe_dataset(Dataset):
    def __init__(self, game_dataset, probe_layer, property_type="new", device="cpu"):
        self.game_dataset = game_dataset
        self.property_type = property_type  # 'old' vs 'new'
        self.property_modifier = np.concatenate([np.ones((1, 64)) * (-1) ** i for i in range(59)], axis=0)
        self.device = device

        # Define the GPT probe model to return activations
        mconf = GPTConfig(game_dataset.vocab_size, game_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
        GPT_probe = GPTforProbing(mconf, probe_layer=probe_layer)
        GPT_probe.load_state_dict(torch.load("../EWOthello/ckpts/gpt_synthetic.ckpt"))
        for param in GPT_probe.parameters():
            param.requires_grad = False
        GPT_probe.eval()
        self.GPT_probe = GPT_probe.to(device)

    def __len__(self):
        return len(self.game_dataset)

    def __getitem__(self, index):
        x, _ = self.game_dataset[index]
        tbf = [self.game_dataset.itos[_] for _ in x.tolist()]
        valid_until = tbf.index(-100) if -100 in tbf else 999

        # Get the board state vectors
        a = OthelloBoardState()
        board_state = a.get_gt(tbf[:valid_until], "get_state")
        if self.property_type == "new":
            board_state = (np.array(board_state) - 1.0) * self.property_modifier[:valid_until, :] + 1.0

        # Get the activation vectors
        act = self.GPT_probe(x[None, :].to(self.device))[0, :valid_until, :].detach().cpu()

        return act, torch.tensor(board_state, dtype=torch.float32)
