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
    MODEL_DIR, LR, SELF_TRAINING_ROUND, L2_REG
)
import logging
from utils.rules import has_won
from utils.database import Collection
from models.dataloader import GameDataset
from cli.train import single_self_play, multiprocessing_selfplay, pool_selfplay
from cli.self_play import train_model, clean_gpu_cache, save

try:
    set_start_method('forkserver',force=True)
except RuntimeError:
    print('faild to spawn mode')
    pass

logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

if __name__ == "__main__":
    logging.info('start training v3')
    cpu = 6
    init_round = 0
    writer_idx = 0
    load_model = None
    round_limit = 1000
    lr_multiplier = 1.0
    skip_first = False
    '''
        load_model: model name to load in string
        cpu: total multiprocessing core to use
        round_limit: total self play round to run
        init_round: initial run, use for checkpoint 
        log_dir: tensorboard log path
    '''
    # clear previous tensorboard log
    name = 'DualResNetv3_0.pt'
    writer_idx = 0
    l2_const = L2_REG
    epochs = 1 # number of epoch training
    batch_size = 256
    num_iter = writer_idx
    n_playout = PLAYOUT_ROUND

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info('Use CUDA')

    model = DualResNet()
    model = model.to(device)
    model = model.share_memory()
    optimizer = optim.Adam(model.parameters(), lr=LR,
                                    weight_decay=l2_const)

    round_count = init_round
    logging.info('Start self play')
    load_model = 'DualResNetv3_{}.pt'.format(round_count)
    while True:
        '''
            Self play through MCTS
        '''
        logging.info('Round : {}'.format(round_count))
        if skip_first:
            skip_first = False
            logging.info('Skipping first round, straight into backprop')
        else:
            model.eval()
            pool_selfplay(model, cpu, rounds=PARALLEL_SELF_PLAY, n_playout=n_playout)

        if round_count > 10 and n_playout < 200:
            n_playout += 5

        if os.path.isfile(os.path.join(MODEL_DIR, load_model)):
            checkpoint = torch.load(os.path.join(MODEL_DIR, load_model))
            model.load_state_dict(checkpoint['network'])
            round_count += 1
            load_model = 'DualResNetv3_{}.pt'.format(round_count)
            