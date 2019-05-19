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

def train_model(model, optimizer, round_count, num_iter, writer, device, epochs=1, batch_size=128, kl_target=0.02,lr_multiplier = 1.0, lr=LR):
    dataloader = torch.utils.data.DataLoader(
        GameDataset('beta', model.VERSION, training_round=SELF_TRAINING_ROUND),
        batch_size=batch_size, shuffle=True)
    loss = {}
    batch_num = len(dataloader) // batch_size
    model.train()
    
    
    test_batch = next(iter(dataloader))
    feature = test_batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)    
    old_softmax, _ = model(feature)
    old_softmax = old_softmax.cpu().detach().numpy()
    kl = 0
    for epoch in range(epochs):
        with tqdm(total=batch_num, ncols=150) as t:
            t.set_description('Epoch %2d/%2d' % (epoch + 1, epochs))
            for batch in dataloader:
                feature = batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)
                mcts_softmax = batch['softmax'].to(device, dtype=torch.float)
                value = batch['value'].to(device, dtype=torch.float)
                optimizer.zero_grad()
                set_learning_rate(optimizer, min(lr*lr_multiplier, 0.01) )

                log_pred_softmax, pred_value = model(feature)
                value_loss, policy_loss, loss = alpha_loss(log_pred_softmax, mcts_softmax, pred_value, value)
                entropy = -torch.mean(torch.sum(torch.exp(log_pred_softmax) * log_pred_softmax, 1))

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
                writer.add_scalar('entropy', entropy, num_iter)
                writer.add_scalar('learning_rate', min(lr*lr_multiplier, 0.01), num_iter)

                new_softmax, _ = model(feature)
                new_softmax_np = new_softmax.cpu().detach().numpy()
                if new_softmax_np.shape == old_softmax.shape:
                    kl = np.mean(np.sum(np.exp(old_softmax) * (
                            (old_softmax + 1e-10) - (new_softmax_np + 1e-10)),
                            axis=1))
                writer.add_scalar('KL_div', kl, num_iter)
                # if kl > kl_target * 4:  # early stopping if D_KL diverges badly
                #     break
            # adaptively adjust the learning rate
        if kl > kl_target * 2 and lr_multiplier > 0.1:
            lr_multiplier /= 1.5
        elif kl < kl_target / 2 and lr_multiplier < 10:
            lr_multiplier *= 1.5

    return model, optimizer, num_iter, lr_multiplier

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
    log_dir='./log/v6.0_%s'
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
    if load_model:
        logging.info('Load checkpoint model')
        checkpoint = torch.load(os.path.join(MODEL_DIR, load_model))
        model.load_state_dict(checkpoint['network'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        round_count = checkpoint['round']+1
        if 'tensorboard_iter' in checkpoint:
            num_iter = checkpoint['tensorboard_iter']
        else:
            shutil.rmtree(log_dir % name, ignore_errors=True)
    else:
        shutil.rmtree(log_dir % name, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir % name)

    logging.info('Start self play')

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

        if round_count > 10 and n_playout < 180:
            n_playout += 5
        '''
            Backpropagation using self play MCTS
        '''
        model, optimizer, num_iter, lr_multiplier = train_model(model, optimizer, round_count, num_iter, writer,device, batch_size=batch_size, lr_multiplier=lr_multiplier)
        save(model, optimizer, round_count, 'DualResNetv3_{}.pt'.format(round_count), num_iter+1)
        round_count += 1
        if round_count > round_limit:
            break