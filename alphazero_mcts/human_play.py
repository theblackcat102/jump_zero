# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from game import Game
from rules import get_all_move
import pickle
from converter import *
from mcts import MCTS as MCTS_Pure
from alpha_mcts import AlphaMCTS
import time
from converter import DualResNetNumpy

with open('numpy_nn.pkl', 'rb') as f:
    models = pickle.load(f)

'''
    Pytorch
'''
from dualresnet import DualResNet
import torch

model = DualResNet()
checkpoint = torch.load(os.path.join('../checkpointv2', 'DualResNetv6.2_154.pt'), map_location='cpu')
model.load_state_dict(checkpoint['network'])
model.eval()

def start_play(player1, player2, board_input,init_playout=60, start_player=1):
    players = {1: player1, 2: player2}
    board_input.current = start_player
    playout_round = init_playout
    '''
    print('Current board:')
        print(board_input.board)
    '''
    while True:
        start_t = time.monotonic()
        print('Curent board:')
        print(board_input.board)
        player_in_turn = players.get(board_input.current)
        player_in_turn.update_with_move(board_input.board)
        move, start, end, eaten = player_in_turn.get_action(board_input.copy())
        player_in_turn.update_with_move(board_input.board)

        print('player {} move: '.format(board_input.current))
        step = [list(start), list(end)]
        print('Start: {}'.format(start))
        print('End: {}'.format(end))
        print('Eaten: {}'.format(eaten))
        if abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1:
            step = get_all_move(board_input.board, start, end, eaten, board_input.current)
        print(step)
        end_t = time.monotonic()
        print('Time taken : {:.4f}'.format(end_t-start_t))
        end, winner, _ = board_input.update_state(move)
        if end:
            print(move)
            if winner == 2:
                print("Game end. Winner is player 2")
                break
            elif winner == 1:
                print("Game end. Winner is player 1")
                break
            elif winner == 3:
                print("Game end. Tie")
                break
        playout_round += 1

def run():
    try:
        game = Game()

        mcts_player_1 = MCTS_Pure(c_puct=5, n_playout=20)
        # mcts_player_2 = MCTS_Pure(c_puct=5, n_playout=30)
        mcts_player_2 = AlphaMCTS(models.policyvalue_function, c_puct=0.2, n_playout=600)

        start_play(mcts_player_1, mcts_player_2, game)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
