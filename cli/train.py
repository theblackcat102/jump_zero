import numpy as np
from models.dualresnet import DualResNet
from utils.game import Game
from pptree import *
from utils.mcts import MCTS
from utils.rules import has_won
from datetime import datetime
import multiprocessing as mp

def self_play(start_color=1):
    # player start at 
    history_stats = {
        'initial': start_color, 
        'mcts_softmax': [],
        'v': 0, # final result
        'time': datetime.now(),
        'board_history': [],
    }
    game = Game(player=start_color)
    model = DualResNet()
    mcts = MCTS(model.policyvalue_function,initial_player=start_color, n_playout=100)
    idx = 0
    initial_temp = 1.0
    temp = initial_temp
    while True:
    # for _ in range(4):
        acts, probability = mcts.get_move_visits(game.copy(), temperature=temp)

        # pick a random move
        max_prob_idx = np.random.choice(len(probability), p=probability) 
        history_stats['board_history'].append(game.board)
        previous_board = np.copy(game.board)
        step, _ = acts[max_prob_idx], probability[max_prob_idx]
        print('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))
        end, winner, reward = game.update_state(step)
        mcts.update_with_move(step)
        if end:
            print('Finish: ')
            print(winner)
            print(reward)
            history_stats['v'] = reward
            # for i in range(10):
            #     print(game.history[i])
            break
        idx += 1
        # an infinitesimal temperature is used, τ → 0
        if idx >= 30 and temp > 1e-3:
            temp /= 10.0
    history_stats['board_history'].append(game.board)
    return history_stats

def multiprocessing_selfplay():
    pool = mp.Pool() 
    res = pool.map(self_play, range(10))
    print(res)

if __name__ == "__main__":
    self_play()