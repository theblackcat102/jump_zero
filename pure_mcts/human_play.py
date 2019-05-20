# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
from game import Game
from mcts import MCTS as MCTS_Pure


def start_play(player1, player2, board_input,init_playout=100, start_player=1):
    players = {1: player1, -1: player2}
    board_input.current = start_player
    playout_round = init_playout
    '''
    print('Current board:')
        print(board_input.board)
    '''
    while True:
        print('Curent player: {}'.format(board_input.current))
        
        player_in_turn = players.get(board_input.current)
        player_in_turn._n_playout = playout_round
        move = player_in_turn.get_action(board_input.copy())
        print('player {} move: '.format(board_input.current))
        print('------------------------------')
        # 
        print(move - board_input.board)
        end, winner, _ = board_input.update_state(move)
        if end:
            if winner == -1:
                print("Game end. Winner is player 1")
                break
            elif winner == 1:
                print("Game end. Winner is player 2")
                break
            elif winner == 2:
                print("Game end. Tie")
                break
        playout_round += 1

def run():
    try:
        game = Game()

        mcts_player_1 = MCTS_Pure(c_puct=5, n_playout=100)

        mcts_player_2 = MCTS_Pure(c_puct=5, n_playout=100)

        start_play(mcts_player_1, mcts_player_2, game)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
