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
    def __init__(self, codebase_version, model_version, training_round=50000 ):
        self.collection = Collection(codebase_version, model_version)
        self.data = []
        logging.info('Start loading data')
        count = 0

        for playout in tqdm(self.collection.get_all()):
            if count > training_round:
                break
            history = [np.zeros((BOARD_WIDTH, BOARD_HEIGHT))]*8
            value = playout['v']
            for idx, current_board in enumerate(playout['board_history']):
                # current_board = playout['board_history'][idx]
                current_player = playout['player_round'][idx]
                inputs = generate_extractor_input(current_board, history, current_player)
                softmax = playout['mcts_softmax'][idx]
                self.data.append({
                    'input': np.array(inputs),
                    'softmax': np.array(softmax),
                    'v': float(value),
                })
                if count > training_round:
                    break
                count += 1
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
        dataset=GameDataset('beta', 'v1.0', training_round=100),
        batch_size=32, shuffle=True, drop_last=True)
    for batch in x_dataloader:
        feature = batch['input'].to(device, dtype=torch.float).permute(0, 3, 1, 2)
        softmax = batch['softmax'].to(device, dtype=torch.float)
        value = batch['value'].to(device, dtype=torch.float)

        pred_softmax, pred_value = model(feature)
        value_loss, policy_loss, loss = alpha_loss(pred_softmax, softmax, pred_value, value)
        print('value: {}, loss {} '.format(value_loss, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

