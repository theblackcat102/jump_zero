import os, sys
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from torch.multiprocessing import Pool, Process, set_start_method, Manager
import torch.optim as optim
import logging
import shutil
import torch
from tqdm import tqdm
from models.dualresnet import DualResNet, alpha_loss, set_learning_rate
from utils.game import Game
from utils.mcts import MCTS
from pure_mcts.mcts import MCTS as PureMCTS
from utils.settings import ( 
    EPS, ALPHA, PLAYOUT_ROUND, PARALLEL_SELF_PLAY, 
    MODEL_DIR, LR, SELF_TRAINING_ROUND, L2_REG, CPU_COUNT
)
import logging
from utils.rules import has_won
from utils.database import Collection
from models.dataloader import GameDataset
from cli.train import single_self_play, multiprocessing_selfplay, pool_selfplay
# os.makedirs(MODEL_DIR, safe=True)

logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.WARNING)


try:
    set_start_method('forkserver',force=True)
except RuntimeError:
    print('faild to spawn mode')
    pass

logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def save(model, optimizer, round_count, model_name, tensorboard_iter):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'network': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'tensorboard_iter': tensorboard_iter,
        'round': round_count}, 
        os.path.join(MODEL_DIR, model_name))

def clean_gpu_cache():
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

def simulation(mcts1, mcts2, start_color=1):
    game = Game(player=start_color)
    # time_limit(900) # cannot time limit when using process pool
    # do not set gradient to zero using with notation
    idx = 0

    logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, game.board ))
    for _ in range(401):

        # Player 1 first action
        step = mcts1.get_action(game.copy(), temp=1)
        logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, step ))
        end, winner, reward = game.update_state(np.copy(step))
        mcts2.update_with_move(step)

        if end:
            if winner == 1:
                logging.info('Player 1 has won!')
            elif winner == -1:
                logging.info('Player 2 has won!')
            else:
                logging.info('Its a Tie')
            break

        # Player 2 take action
        step = mcts2.get_action(game.copy(), temp=1)
        logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, step ))
        end, winner, reward = game.update_state(np.copy(step))
        mcts1.update_with_move(step)

        if end:
            if winner == 1:
                logging.info('Player 1 has won!')
            elif winner == -1:
                logging.info('Player 2 has won!')
            else:
                logging.info('Its a Tie')
            break

        idx += 1



# train_pool = Pool()

if __name__ == "__main__":
    logging.info('start self play mode')
    # log_dir='./log/v6.1_%s'
    check_point_dir = './v1.60'
    # check_point_dir = './checkpointv2'
    load_model1 = 'DualResNetv3_69.pt' #'DualResNetv3_14.pt'
    load_model2 = 'DualResNetv3_52.pt' #'DualResNetv3_14.pt'
    # clear previous tensorboard log

    model1 = DualResNet()
    logging.info('Load checkpoint model')
    checkpoint = torch.load(os.path.join(check_point_dir, load_model1), map_location='cpu')
    model1.load_state_dict(checkpoint['network'])
    model1.eval()

    model2 = DualResNet()
    checkpoint = torch.load(os.path.join(check_point_dir, load_model2), map_location='cpu')
    model2.load_state_dict(checkpoint['network'])
    model2.eval()

    logging.info('Start self play')
    n_playout = 280
    mcts1 = MCTS(model1.policyvalue_function, n_playout=n_playout, self_play=False)
    mcts2 = MCTS(model2.policyvalue_function, n_playout=n_playout, self_play=False)
    pure_mcts = PureMCTS(n_playout=40)
    simulation(mcts1, pure_mcts)
