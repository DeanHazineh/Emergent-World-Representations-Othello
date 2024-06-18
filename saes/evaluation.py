import torch
from sae import SAETemplate
from board_states import get_board_states
from tqdm import tqdm
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAUROC

def compute_all_aurocs(sae: SAETemplate, evaluation_dataset:DataLoader, alternate_players=True):
    '''
    computes aurocs of each sae feature on the entire evaluation_dataset
    returns a shape (N,64,3) tensor, where N is the number of features
    '''
    _, hidden_layers, __=sae.catenate_outputs_on_dataset(evaluation_dataset, include_loss=False)
    board_states= get_board_states(evaluation_dataset,alternate_players=alternate_players)
    board_states=sae.trim_to_window(board_states)
    hidden_layers=hidden_layers.flatten(end_dim=-2)
    board_states=board_states.flatten(end_dim=-2)
    game_not_ended_mask=board_states[:,0]>-100
    hidden_layers=hidden_layers[game_not_ended_mask]
    board_states=board_states[game_not_ended_mask]
    aurocs=torch.zeros((hidden_layers.shape[1], board_states.shape[1], 3))
    for i, feature_activation in tqdm(enumerate(hidden_layers.transpose(0,1))):
        for j, board_position in enumerate(board_states.transpose(0,1)):
            for k, piece_class in enumerate([0,1,2]):
                is_target_piece=board_position==piece_class
                ended_game_mask= board_position>-100
                metric = BinaryAUROC()
                metric.update(feature_activation[ended_game_mask], is_target_piece[ended_game_mask].int())
                aurocs[i,j,k]=float(metric.compute())
    return aurocs
