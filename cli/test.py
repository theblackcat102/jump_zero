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

    mcts2.get_action(game.copy())
    mcts1.get_action(game.copy())

    logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, game.board ))    
    for _ in range(401):

        # Player 1 first action
        step = mcts1.get_action(game.copy())
        logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, step ))
        end, winner, reward = game.update_state(np.copy(step))
        mcts2.update_with_move(step)
        # print('Updated board:\n{}'.format(game.board - step))
        if end:
            logging.info('Player 1 has won!')
            break

        # Player 2 take action
        step = mcts2.get_action(game.copy())
        logging.info('Step: {}, current player: {}\nBoard: \n{}'.format(idx, game.current, step ))
        end, winner, reward = game.update_state(np.copy(step))
        mcts1.update_with_move(step)
        # print('Updated board:\n{}'.format(game.board - step))
        if end:
            logging.info('Player 2 has won!')
            break
        idx += 1
        if idx > 3:
            mcts1._n_playout = 64
            mcts2._n_playout = 64



# train_pool = Pool()

if __name__ == "__main__":
    logging.info('start training v3')
    # model_name = 'DualResNet_2.pt'
    # train_selfplay(load_model=None, 
    #     cpu=10, init_round=0, log_dir='./log/v3_%s', 
    #     skip_first=False)
    cpu = CPU_COUNT
    init_round = 0
    writer_idx = 0
    log_dir='./log/v6.1_%s'
    load_model1 = 'DualResNetv3_16.pt' #'DualResNetv3_14.pt'
    load_model2 = 'DualResNetv3-1_6.pt' #'DualResNetv3_14.pt'
    round_limit = 1000
    lr_multiplier = 1.0 # default =1
    skip_first = False
    '''
        load_model: model name to load in string
        cpu: total multiprocessing core to use
        round_limit: total self play round to run
        init_round: initial run, use for checkpoint 
        log_dir: tensorboard log path
    '''
    # clear previous tensorboard log
    n_playout = 64
    num_iter = writer_idx
    n_playout = PLAYOUT_ROUND

    round_count = init_round

    model1 = DualResNet()
    logging.info('Load checkpoint model')
    checkpoint = torch.load(os.path.join('./v1.60', load_model1), map_location='cpu')
    model1.load_state_dict(checkpoint['network'])

    model2 = DualResNet()
    checkpoint = torch.load(os.path.join('./v1.60', load_model2), map_location='cpu')
    model2.load_state_dict(checkpoint['network'])


    logging.info('Start self play')

    mcts1 = MCTS(model1.policyvalue_function, n_playout=n_playout, self_play=False)
    mcts2 = MCTS(model2.policyvalue_function, n_playout=n_playout, self_play=False)

    simulation(mcts1, mcts2)