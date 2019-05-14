import time
import numpy as np
from datetime import datetime
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import logging
import torch
from tqdm import tqdm
from models.dualresnet import DualResNet
from utils.game import Game
from utils.mcts import MCTS
from utils.settings import EPS, ALPHA, TEMPERATURE_MOVE, PROCESS_TIMEOUT, PLAYOUT_ROUND
from utils.rules import has_won
from utils.database import Collection

try:
    set_start_method('spawn',force=True)
except RuntimeError:
    print('faild to spawn mode')
    pass
logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def single_self_play(process_rank, model, return_dict=None, start_color=1, n_playout=PLAYOUT_ROUND):
    # player start at 
    model.eval()
    history_stats = {
        'total_steps':0,
        'initial': start_color, 
        'mcts_softmax': [],
        'player_round': [],
        'v': 0, # final result
        'time': datetime.now(),
        'board_history': [],
    }
    game = Game(player=start_color)

    mcts = MCTS(model.policyvalue_function, initial_player=start_color, n_playout=n_playout)
    idx = 0
    initial_temp = 1.0
    temp = initial_temp
    while True:
        acts, probability, mcts_softmax = mcts.get_move_visits(game.copy(), temperature=temp)
        # pick a random move
        valid_move_count = len(probability)
        move_idx = np.random.choice(valid_move_count, 
            p=(1-EPS)*probability + EPS*np.random.dirichlet(ALPHA*np.ones(valid_move_count))
        )
        step = acts[move_idx]
        history_stats['player_round'].append(game.current)

        # logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))

        end, winner, reward = game.update_state(step)

        history_stats['board_history'].append(game.board)
        history_stats['mcts_softmax'].append(mcts_softmax)

        mcts.update_with_move(step)

        if end:
            history_stats['v'] = reward
            history_stats['winner'] = winner
            # for i in range(10):
            #     print(game.history[i])
            break
        idx += 1
        # an infinitesimal temperature is used, τ → 0
        if idx >= TEMPERATURE_MOVE and temp > 1e-3:
            temp /= 10.0
    # add the final board result
    history_stats['board_history'].append(game.board)
    history_stats['total_steps'] = idx
    if return_dict:
        return_dict[process_rank] = history_stats
    collection = Collection('beta', model.VERSION)
    collection.add_game(history_stats)


def multiprocessing_selfplay(model, cpu=5):
    logging.debug('Start parallel self play')
    processes = []
    for rank in range(cpu):
        p = Process(target=single_self_play, args=(rank, model, None))
        p.start()
        processes.append(p)
    try:
        for p in processes:
            p.join()
    except:
        logging.debug('failed process')

def kill_process(pool):
    time.sleep( 600 )
    pool.terminate()

def pool_selfplay(model, cpu=5, rounds=100):
    # logging.debug('Start parallel self play')
    pool = Pool(processes=cpu)
    processes = []
    for i in range(rounds):
        res = pool.apply_async(single_self_play, (i, model, None))
        processes.append(res)
    for proc in tqdm(processes, total=rounds):
        proc.wait(timeout=1200)
        time.sleep(3)
    pool.close()
    pool.join()

if __name__ == "__main__":
    model = DualResNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    collection = Collection('beta', model.VERSION)
    logging.info('Start selfplay')
    # multiprocessing_selfplay(model, cpu=5)
    # return_dict = {}
    game_stat = single_self_play(1, model, None)
    # collection.add_batch(return_dict.values())
    logging.info('End self play')
