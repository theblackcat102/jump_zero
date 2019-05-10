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
from models.dualresnet import DualResNet, alpha_loss
from utils.game import Game
from utils.mcts import MCTS
from utils.settings import EPS, ALPHA, PLAYOUT_ROUND, PARALLEL_SELF_PLAY, MODEL_DIR, LR
from utils.rules import has_won
from utils.database import Collection
from models.dataloader import GameDataset
from cli.train import self_play, multiprocessing_selfplay
# os.makedirs(MODEL_DIR, safe=True)
try:
    set_start_method('spawn',force=True)
except RuntimeError:
    print('faild to spawn mode')
    pass


logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.DEBUG)

def save(model, optimizer, round_count, model_name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        'network': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'round': round_count}, 
        os.path.join(MODEL_DIR, model_name))


def train_selfplay(load_model=None, cpu = 10, rounds=100, log_dir='./log/%s'):
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)
    model = DualResNet()
    l2_const = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=LR,
                                    weight_decay=l2_const)
    round_count = 0
    if load_model:
        checkpoint = torch.load(os.path.join(MODEL_DIR, load_model))
        model.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        round_count = checkpoint['round']
        
    if torch.cuda.is_available():
        logging.info('Use CUDA')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logging.info('Start self play')
    epochs = 1
    batch_size = 128
    collection = Collection('beta', model.VERSION)
    while True:
        logging.info('Round : {}'.format(round_count))
        parallel_iter = PARALLEL_SELF_PLAY//int(cpu)
        for iter in tqdm(range(parallel_iter), total=parallel_iter):
            game_stats = multiprocessing_selfplay(model, cpu)
            # print(len(game_stats))
            result = collection.add_batch(game_stats)
            # print(result)

        dataloader = torch.utils.data.DataLoader(
            GameDataset('beta', model.VERSION, training_round=30),
            batch_size=batch_size, shuffle=True, drop_last=True)
        loss = {}
        batch_num = len(dataloader) // batch_size
        for epoch in range(epochs):
            with tqdm(total=batch_num, ncols=150) as t:
                t.set_description('Epoch %2d/%2d' % (epoch + 1, epochs))
                for batch in dataloader:
                    feature = batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)
                    softmax = batch['softmax'].to(device, dtype=torch.float)
                    value = batch['value'].to(device, dtype=torch.float)

                    pred_softmax, pred_value = model(feature)
                    value_loss, policy_loss, loss = alpha_loss(pred_softmax, softmax, pred_value, value)

                    t.update(1)
                    t.set_postfix(categorical='%.4f' % policy_loss,
                                value='%.4f' % value_loss,
                                total='%.4f' % loss)
        print('Saving model...')
        save(model, optimizer, round_count, 'DualResNet')
        round_count += 1
        if round_count > rounds:
            break

if __name__ == "__main__":
    logging.info('start training')
    train_selfplay(cpu=13)