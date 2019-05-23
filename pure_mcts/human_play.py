# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from game import Game
from rules import get_all_move
from mcts import MCTS as MCTS_Pure


def start_play(player1, player2, board_input,init_playout=60, start_player=1):
    players = {1: player1, 2: player2}
    board_input.current = start_player
    playout_round = init_playout
    '''
    print('Current board:')
        print(board_input.board)
    '''
    while True:
        print('Curent board:')
        print(board_input.board)
        player_in_turn = players.get(board_input.current)
        move, start, end, eaten = player_in_turn.get_action(board_input.copy())
        player_in_turn.update_with_move(-1)
        print('player {} move: '.format(board_input.current))
        step = [list(start), list(end)]
        print('Start: {}'.format(start))
        print('End: {}'.format(end))
        print('Eaten: {}'.format(eaten))
        if abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1:
            step = get_all_move(board_input.board, start, end, eaten, board_input.current)
        print(step)
        end, winner, _ = board_input.update_state(move)
        if end:
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

        mcts_player_1 = MCTS_Pure(c_puct=5, n_playout=30)

        mcts_player_2 = MCTS_Pure(c_puct=5, n_playout=30)

        start_play(mcts_player_1, mcts_player_2, game)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
