import numpy as np
from models.dualresnet import DualResNet
from utils.game import Game
from pptree import *
from utils.mcts import MCTS
from utils.rules import has_won

def self_play(start_color=-1):
    # player start at 
    game = Game(player=start_color)
    model = DualResNet()
    mcts = MCTS(model.policyvalue_function,initial_player=start_color, n_playout=100)
    idx = 0
    while True:

        acts, probability = mcts.get_move_visits(game.copy())
        max_prob_idx = np.argmax(probability)
        previous_board = np.copy(game.board)
        step, _ = acts[max_prob_idx], probability[max_prob_idx]
        print('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))
        end, winner, reward = game.update_state(step)
        mcts.update_with_move(step)
        if end:
            print('Finish: ')
            print(winner)
            print(reward)
            # for i in range(10):
            #     print(game.history[i])
            break
        idx += 1

if __name__ == "__main__":
    self_play()