import numpy as np
from datetime import datetime
import multiprocessing as mp
import logging

from models.dualresnet import DualResNet
from utils.game import Game
from utils.mcts import MCTS
from utils.settings import EPS, ALPHA
from utils.rules import has_won
from utils.database import Collection


logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def self_play(model, start_color=1, n_playout=100):
    # player start at 
    history_stats = {
        'total_steps':0,
        'initial': start_color, 
        'mcts_softmax': [],
        'v': 0, # final result
        'time': datetime.now(),
        'board_history': [],
    }
    game = Game(player=start_color)

    mcts = MCTS(model.policyvalue_function,initial_player=start_color, n_playout=n_playout)
    idx = 0
    initial_temp = 1.0
    temp = initial_temp
    while True:
    # for _ in range(4):
        acts, probability, mcts_softmax = mcts.get_move_visits(game.copy(), temperature=temp)
        # pick a random move
        valid_move_count = len(probability)
        move_idx = np.random.choice(valid_move_count, 
            p=(1-EPS)*probability + EPS*np.random.dirichlet(ALPHA*np.ones(valid_move_count))
        )
        previous_board = np.copy(game.board)
        step = acts[move_idx]

        logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))
        logging.debug(mcts_softmax)
        end, winner, reward = game.update_state(step)

        history_stats['board_history'].append(game.board.tolist())
        history_stats['mcts_softmax'].append(mcts_softmax.tolist())

        mcts.update_with_move(step)
        if end:
            history_stats['v'] = reward
            history_stats['winner'] = winner
            logging.debug('Finish: ')
            logging.debug(winner)
            logging.debug(reward)
            logging.debug(game.board)
            # for i in range(10):
            #     print(game.history[i])
            break
        idx += 1
        # an infinitesimal temperature is used, τ → 0
        if idx >= 30 and temp > 1e-3:
            temp /= 10.0
    # add the final board result
    history_stats['board_history'].append(game.board.tolist())
    history_stats['total_steps'] = idx
    return history_stats

def multiprocessing_selfplay(model):
    logging.info('Start parallel self play')
    pool = mp.Pool(10) 
    res = pool.map(self_play, [model for idx in range(10)])
    logging.info('End parallel self play')
    print(res)
    return res

if __name__ == "__main__":
    model = DualResNet()
    collection = Collection('beta', model.VERSION)
    logging.info('Start selfplay')
    # game_stats = multiprocessing_selfplay(model)
    game_stat = self_play(model)
    collection.add_batch([game_stat])
    logging.info('End self play')
