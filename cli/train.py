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
from utils.system_limit import memory_limit, time_limit, get_gpu_memory_map

logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def single_self_play(process_rank, model, n_playout=PLAYOUT_ROUND, start_color=1 ):
    # quick fix to kill memory leak process from pytorch gpu api
    idx = 0
    initial_temp = 1.0
    temp = initial_temp
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
    try:
        memory_limit(25*1024*1024*1024)
        # time_limit(900) # cannot time limit when using process pool
        # do not set gradient to zero using with notation
        mcts = MCTS(model.policyvalue_function, initial_player=start_color, n_playout=n_playout)
        for _ in range(401):
            acts, probability, mcts_softmax = mcts.get_move_visits(game.copy(), temperature=temp)
            # pick a random move
            valid_move_count = len(probability)
            move_idx = np.random.choice(valid_move_count, 
                p=(1-EPS)*probability + EPS*np.random.dirichlet(ALPHA*np.ones(valid_move_count))
            )
            step = acts[move_idx]
            history_stats['player_round'].append(game.current)
            # logging.info('Step: {}, current player: {}\n'.format(idx, game.current ))
            # logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, previous_board-step ))
            history_stats['board_history'].append(game.board)
            history_stats['mcts_softmax'].append(mcts_softmax)

            # update board, switch user perspective, check game state 
            end, winner, reward = game.update_state(step)
            # change root node to current child
            mcts.update_with_move(step)
            history_stats['v'] = reward
            history_stats['winner'] = winner
            if end:
                break
            idx += 1
            # an infinitesimal temperature is used, τ → 0
            if idx >= TEMPERATURE_MOVE and temp > 1e-3:
                temp /= 10.0
            # add the final board result
            history_stats['board_history'].append(game.board)
            history_stats['total_steps'] = idx

        collection = Collection('beta', model.VERSION)
        collection.add_game(history_stats)
    except ValueError:
        logging.warning('value error')
    except TimeoutError:
        logging.warning('timeout error')
    except:
        logging.warning('memory error')
    finally:
        return False
    return True


def multiprocessing_selfplay(model, cpu=5):
    logging.debug('Start parallel self play')
    processes = []
    model = model.to(torch.device('cuda'))

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

def pool_selfplay(model, cpu=5, rounds=100, n_playout=PLAYOUT_ROUND):
    # logging.debug('Start parallel self play')
    with Pool(processes=cpu) as pool:
        processes = []
        for i in range(rounds):
            res = pool.apply_async(single_self_play, (i, model, n_playout))
            processes.append(res)
        success_count = 0
        for proc in tqdm(processes, total=rounds):
            try:
                proc.wait()
                success = proc.get()
                if success:
                    success_count += 1
            except KeyboardInterrupt:
                return 0
            except BaseException as e:
                logging.warning('failed self play {}'.format(str(e)))
    print("{}/{} self play has successfully done".format(success_count, rounds))

if __name__ == "__main__":
    model = DualResNet()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    collection = Collection('beta', model.VERSION)
    logging.info('Start selfplay')
    game_stat = single_self_play(1, model, n_playout=120)
    logging.info('End self play')
