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
from utils.settings import ( 
    EPS, ALPHA, PLAYOUT_ROUND, PARALLEL_SELF_PLAY, 
    MODEL_DIR, LR, SELF_TRAINING_ROUND, L2_REG
)
from utils.rules import has_won
from utils.database import Collection
from models.dataloader import GameDataset
from cli.train import single_self_play, multiprocessing_selfplay, pool_selfplay
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

def clean_gpu_cache():
    num_gpus = torch.cuda.device_count()
    for gpu_id in range(num_gpus):
        torch.cuda.set_device(gpu_id)
        torch.cuda.empty_cache()

def train_model(model, optimizer, round_count, num_iter, writer, epochs=1, batch_size=128):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = torch.utils.data.DataLoader(
        GameDataset('beta', model.VERSION, training_round=SELF_TRAINING_ROUND),
        batch_size=batch_size, shuffle=True)
    loss = {}
    batch_num = len(dataloader) // batch_size
    model = model.cuda()
    model.train()
    for epoch in range(epochs):
        with tqdm(total=batch_num, ncols=150) as t:
            t.set_description('Epoch %2d/%2d' % (epoch + 1, epochs))
            for batch in dataloader:
                feature = batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)
                mcts_softmax = batch['softmax'].to(device, dtype=torch.float)
                value = batch['value'].to(device, dtype=torch.float)
                optimizer.zero_grad()
                pred_softmax, pred_value = model(feature)
                value_loss, policy_loss, loss = alpha_loss(pred_softmax, mcts_softmax, pred_value, value)
                loss.backward()
                optimizer.step()

                t.update(1)
                num_iter += 1
                t.set_postfix(categorical='%.4f' % policy_loss,
                            value='%.4f' % value_loss,
                            total='%.4f' % loss)
                writer.add_scalar('categorical_loss', policy_loss, num_iter)
                writer.add_scalar('value_loss', value_loss, num_iter)
                writer.add_scalar('total_loss', loss, num_iter)
    # print('Saving model...')
    save(model, optimizer, round_count, 'DualResNetv2_{}.pt'.format(round_count))
    clean_gpu_cache()
    return model, optimizer, num_iter

# train_pool = Pool()

if __name__ == "__main__":
    logging.info('start training')
    # model_name = 'DualResNet_2.pt'
    # train_selfplay(load_model=None, 
    #     cpu=10, init_round=0, log_dir='./log/v3_%s', 
    #     skip_first=False)
    cpu = 12
    init_round = 1
    log_dir='./log/v3.1_%s'
    load_model = 'DualResNetv2_0.pt'
    round_limit = 1000
    skip_first = False
    '''
        load_model: model name to load in string
        cpu: total multiprocessing core to use
        round_limit: total self play round to run
        init_round: initial run, use for checkpoint 
        log_dir: tensorboard log path
    '''
    # clear previous tensorboard log
    name = load_model if load_model else 'DualResNet'
    shutil.rmtree(log_dir % name, ignore_errors=True)
    l2_const = L2_REG
    epochs = 1 # number of epoch training
    batch_size = 64
    num_iter = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info('Use CUDA')

    writer = SummaryWriter(log_dir=log_dir % name)
    model = DualResNet()
    model = model.to(device)
    model = model.share_memory()
    optimizer = optim.Adam(model.parameters(), lr=LR,
                                    weight_decay=l2_const)

    round_count = init_round
    if load_model:
        logging.info('Load checkpoint model')
        checkpoint = torch.load(os.path.join(MODEL_DIR, load_model))
        model.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        round_count = checkpoint['round']+1

    logging.info('Start self play')
    # collection = Collection('beta', model.VERSION)

    while True:
        '''
            Self play through MCTS
        '''
        logging.info('Round : {}'.format(round_count))
        if skip_first:
            skip_first = False
            logging.info('Skipping first round, straight into backprop')
        else:
            parallel_iter = PARALLEL_SELF_PLAY//int(cpu)
            # for _ in tqdm(range(parallel_iter), total=parallel_iter):
            #     # single_self_play(1, model)
            #     multiprocessing_selfplay(model, cpu)
            #     # _ = collection.add_batch(game_stats)
            pool_selfplay(model, cpu, PARALLEL_SELF_PLAY)
        '''
            Backpropagation using self play MCTS
        '''

        # res = train_pool.apply_async(train_model, (model, optimizer, round_count, num_iter, writer, batch_size,))
        # model, optimizer, num_iter = res.get()
        model, optimizer, num_iter = train_model(model, optimizer, round_count, num_iter, writer, batch_size=batch_size)
        round_count += 1
        if round_count > round_limit:
            break