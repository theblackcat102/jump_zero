import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
import logging
import pymongo
from torch.utils.data import Dataset
from utils.settings import BOARD_WIDTH, BOARD_HEIGHT
from utils.database import Collection
import logging
from utils.rules import generate_extractor_input

logging.basicConfig(format='%(asctime)s:%(message)s',level=logging.INFO)

class GameDataset(Dataset):
    def __init__(self, codebase_version, model_version, training_round=500, balance_value=True):
        self.collection = Collection(codebase_version, model_version)
        self.data = []

        logging.info('Start loading data')
        count = 0
        player_win, tie, opponent_win = 0,  0,  0
        avg_rounds = 0
        win_data, lose_data, tie_data = [], [], []
        for playout in tqdm(self.collection.get_all()):
            if min([player_win, opponent_win, tie ]) > (training_round / 3) and balance_value:
                break
            if not balance_value and (player_win+opponent_win+tie) > training_round:
                break
            history = [np.zeros((BOARD_WIDTH, BOARD_HEIGHT))]*8
            value = playout['v']
            if value > 0.5:
                player_win += 1
            elif value < -0.5:
                opponent_win += 1
            elif value == 0:
                tie += 1
            current_player = playout['player_round'][0]
            avg_rounds += len(playout['mcts_softmax'])
            for idx, softmax in enumerate(playout['mcts_softmax']):
                current_board = playout['board_history'][idx]
                inputs = generate_extractor_input(current_board, history, current_player)
                current_player = -1 if current_player==1 else -1
                # softmax = playout['mcts_softmax'][idx]
                if balance_value:
                    if value == 1:
                        win_data.append({
                            'input': np.array(inputs),
                            'softmax': np.array(softmax),
                            'v': float(value),
                        })
                    elif value == -1:
                        lose_data.append({
                            'input': np.array(inputs),
                            'softmax': np.array(softmax),
                            'v': float(value),
                        })
                    else:
                        tie_data.append({
                            'input': np.array(inputs),
                            'softmax': np.array(softmax),
                            'v': float(value),
                        })
                else:
                    self.data.append({
                            'input': np.array(inputs),
                            'softmax': np.array(softmax),
                            'v': float(value),
                        })
            count += 1
        # balance the rounds of winning and losing
        if balance_value:
            for idx in range(min([len(win_data),len(lose_data) ])):
                if len(win_data) > idx:
                    self.data.append( win_data[idx] )
                if len(lose_data) > idx:
                    self.data.append( lose_data[idx] )
                if len(tie_data) > idx:
                    self.data.append( tie_data[idx] )

        logging.info("Average rounds : {0:.4f}".format(avg_rounds/count))
        logging.info("Win {}, Lose {}, Tie {}".format(player_win, opponent_win, tie))

        self.size = len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        return {'input': data['input'], 'softmax': data['softmax'], 'value': data['v']}

    def __len__(self):
        return self.size



if __name__ == "__main__":
    import torch
    import torch.optim as optim
    from models.dualresnet import DualResNet, alpha_loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualResNet()
    model = model.to(device)
    l2_const = 1e-4
    optimizer = optim.Adam(model.parameters(),
                                    weight_decay=l2_const)

    x_dataloader = torch.utils.data.DataLoader(
        dataset=GameDataset('beta', DualResNet.VERSION, training_round=20),
        batch_size=8, shuffle=True, drop_last=True)
    epochs = 1
    epoch=0
    with tqdm(total=len(x_dataloader)//8, ncols=150) as t:
        t.set_description('Epoch %2d/%2d' % (epoch + 1, epochs))
        for batch in tqdm(x_dataloader, total=len(x_dataloader)//128):
            feature = batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)
            softmax = batch['softmax'].to(device, dtype=torch.float)
            value = batch['value'].to(device, dtype=torch.float)
            model.train(); optimizer.zero_grad()
            pred_softmax, pred_value = model(feature)
            value_loss, policy_loss, loss = alpha_loss(pred_softmax, softmax, pred_value, value)
            # print('value: {}, loss {} \n'.format(value_loss, loss))
            t.update(1)
            t.set_postfix(categorical='%.4f' % policy_loss,
                        value='%.4f' % value_loss,
                        total='%.4f' % loss)
            loss.backward()
            optimizer.step()

