import torch

from EWOthello.data.othello import OthelloBoardState
from EWOthello.mingpt.dataset import CharDataset


static_itos = {
    0: -100,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    26: 25,
    27: 26,
    28: 29,
    29: 30,
    30: 31,
    31: 32,
    32: 33,
    33: 34,
    34: 37,
    35: 38,
    36: 39,
    37: 40,
    38: 41,
    39: 42,
    40: 43,
    41: 44,
    42: 45,
    43: 46,
    44: 47,
    45: 48,
    46: 49,
    47: 50,
    48: 51,
    49: 52,
    50: 53,
    51: 54,
    52: 55,
    53: 56,
    54: 57,
    55: 58,
    56: 59,
    57: 60,
    58: 61,
    59: 62,
    60: 63,
}

static_stoi = {
    -100: 0,
    -1: 0,
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
    10: 11,
    11: 12,
    12: 13,
    13: 14,
    14: 15,
    15: 16,
    16: 17,
    17: 18,
    18: 19,
    19: 20,
    20: 21,
    21: 22,
    22: 23,
    23: 24,
    24: 25,
    25: 26,
    26: 27,
    29: 28,
    30: 29,
    31: 30,
    32: 31,
    33: 32,
    34: 33,
    37: 34,
    38: 35,
    39: 36,
    40: 37,
    41: 38,
    42: 39,
    43: 40,
    44: 41,
    45: 42,
    46: 43,
    47: 44,
    48: 45,
    49: 46,
    50: 47,
    51: 48,
    52: 49,
    53: 50,
    54: 51,
    55: 52,
    56: 53,
    57: 54,
    58: 55,
    59: 56,
    60: 57,
    61: 58,
    62: 59,
    63: 60,
}

def report_board_state(board:OthelloBoardState, alternate_players=True):
    '''
    returns a shape (64) tensor of integers listing the current board shape
    coding:
        if alternate_players: 0=enemy piece, 1=blank, 2=active player's piece
        else: 3=white, 4=blank, 5=black
    '''
    board_as_tensor= torch.tensor(board.state, dtype=int) # starts -1 to 1 
    if alternate_players:
        if board.next_hand_color:
            board_as_tensor*=-1 # still -1 to 1
        board_as_tensor+=1 # 0 to 2
    else:
        board_as_tensor+=4 # 3 to 5
    board_as_tensor=board_as_tensor.flatten()
    return board_as_tensor.tolist()


def get_board_states(game_histories:CharDataset, alternate_players=True):
    '''
    converts a (2D) batch of game histories into a (3D) batch of board states. 
    values are recorded *before* the player's turn, so the first board will always be the initial configuration, the second board will be after a single move, etc
    input:
        game_histories: shape (B, W) tensor, where B is batch size and W is window length, as if an entry of a CharDataset. entries will be ints numbered 1-60
        alternate_players: determines whether boards are reported as black/white or own/enemy
    output:
        board_states: shape (B, W, 64) tensor of ints in {0,1,2, -100} or {3,4,5, -100}. coding:
            if alternate_players: 0=enemy piece, 1=blank, 2=active player's piece, -100 = game ended
            else: 3=white, 4=blank, 5=black, -100 = game ended
    '''
    to_return=[]
    for game, _ in game_histories:
        board_history=[]
        board=OthelloBoardState()
        game_as_positions=[static_itos[int(idx)] for idx in game]
        for move in game_as_positions:
            current_board_state=report_board_state(board, alternate_players=alternate_players)
            board_history.append(current_board_state)
            if move==-100: #move indicating game end
                while len(board_history)<len(game_as_positions):
                    board_history.append([-100 for _ in range(64)])
                break
            else:
                board.update([move])
        to_return.append(board_history)
    to_return=torch.tensor(to_return)
    return to_return