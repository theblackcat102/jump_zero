import numpy as np
from datetime import datetime
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import logging
import torch
from models.dualresnet import DualResNet
from utils.game import Game
from utils.mcts import MCTS
from utils.settings import EPS, ALPHA
from utils.rules import has_won
from utils.database import Collection

try:
    set_start_method('spawn',force=True)
except RuntimeError:
    print('faild to spawn mode')
    pass
logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def self_play(process_rank, model, return_dict, start_color=1, n_playout=100):
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

        # logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))

        end, winner, reward = game.update_state(step)

        history_stats['board_history'].append(game.board.tolist())
        history_stats['mcts_softmax'].append(mcts_softmax.tolist())

        mcts.update_with_move(step)
        if end:
            history_stats['v'] = reward
            history_stats['winner'] = winner
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
    return_dict[process_rank] = history_stats
    return history_stats

def multiprocessing_selfplay(model, cpu=5):
    logging.debug('Start parallel self play')
    # pool = Pool(cpu)
    # models = [ model for i in range(int(cpu*CPU_MULTIPLIER)) ]
    # res = pool.map(self_play, models)
    manager = Manager()
    return_dict = manager.dict()
    model.share_memory()
    processes = []
    for rank in range(cpu):
        p = Process(target=self_play, args=(rank, model, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    return [ game for game in return_dict.values() ]

if __name__ == "__main__":
    model = DualResNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    collection = Collection('beta', model.VERSION)
    logging.info('Start selfplay')
    game_stats = multiprocessing_selfplay(model, cpu=5)
    print('Finish {}'.format(len(game_stats)))
    collection.add_batch(game_stats)
    # return_dict = {}
    # game_stat = self_play(1, model, return_dict)
    # collection.add_batch(return_dict.values())
    logging.info('End self play')
