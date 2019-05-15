import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.settings import *

class BasicBlock(nn.Module):
    """
    Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """ 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out

class Extractor(nn.Module):
    """
    This network is used as a feature extractor, takes as input the 'state' defined in
    the AlphaGo Zero paper
    - The state of the past n turns of the board (7 in the paper) for each player.
      This means that the first n matrices of the input state will be 1 and 0, where 1
      is a stone. 
      This is done to take into consideration Go rules (repetitions are forbidden) and
      give a sense of time
    - The color of the stone that is next to play. This could have been a single bit, but
      for implementation purposes, it is actually expended to the whole matrix size.
      If it is black turn, then the last matrix of the input state will be a NxN matrix
      full of 1, where N is the size of the board, 19 in the case of AlphaGo.
      This is done to take into consideration the komi.
    The ouput is a series of feature maps that retains the meaningful informations
    contained in the input state in order to make a good prediction on both which is more
    likely to win the game from the current state, and also which move is the best one to
    make. 
    """

    def __init__(self, inplanes=INPLANE, outplanes=OUTPLANES_MAP):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, stride=1, kernel_size=3,
                        padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)

        for block in range(BLOCKS):
            setattr(self, "res{}".format(block), \
                BasicBlock(outplanes, outplanes))
    

    def forward(self, x):
        """
        x : tensor representing the state
        feature_maps : result of the residual layers forward pass
        """

        x = F.relu(self.bn1(self.conv1(x)))
        for block in range(BLOCKS - 1):
            x = getattr(self, "res{}".format(block))(x)
        
        feature_maps = getattr(self, "res{}".format(BLOCKS - 1))(x)
        return feature_maps

class PolicyNet(nn.Module):
    """
    This network is used in order to predict which move has the best potential to lead to a win
    given the same 'state' described in the Feature Extractor model.
    outplanes : the possible place to put chess
    inplanes : input feature
    """
    def __init__(self, inplanes=OUTPLANES_MAP, outplanes=OUTPLANES):
        super(PolicyNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(outplanes, outplanes*outplanes)

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
 
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes )
        x = self.fc(x)
        probas = self.softmax(x)

        return probas


class ValueNet(nn.Module):

    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1. 
    """

    def __init__(self, inplanes=OUTPLANES_MAP, outplanes=OUTPLANES):
        super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(outplanes, 256)
        self.fc2 = nn.Linear(256, 1)
        

    def forward(self, x):
        """
        x : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
 
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(-1, self.outplanes )
        x = F.relu(self.fc1(x))
        winning = torch.tanh(self.fc2(x))
        return winning

class DualResNet(nn.Module):
    VERSION = 'v1.02'
    def __init__(self, input_place=INPLANE, extractor_output=OUTPLANES_MAP,outputplane=OUTPLANES):
        super(DualResNet, self).__init__()
        self.extractor = Extractor(input_place, extractor_output)
        self.policy_model = PolicyNet( extractor_output, outputplane )
        self.value_model = ValueNet(extractor_output, outputplane )
        self.device = None

    def forward(self, x):
        feature = self.extractor(x)
        softmax_output = self.policy_model(feature)
        prediction = self.value_model(feature)
        return softmax_output, prediction

    def policyvalue_function(self, game):
        feature_input = np.array([game.current_state()])
        if self.device is None:
            self.device = next(self.parameters()).device

        feature_input = torch.from_numpy(feature_input).type('torch.FloatTensor').to(self.device)

        output, value_pred = self.forward(Variable(feature_input))
        probability = output.data.cpu().numpy()[0].reshape(BOARD_WIDTH, BOARD_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT)
        prediction = value_pred.data.cpu().numpy()[0][0]
        del output, value_pred, feature_input
        # we need to convert probability to a board(matrix) and probability(float)

        legal_moves = game.legal_move()
        actions = []
        for (next_board, start_point, end_point ) in legal_moves:
            prob = probability[start_point[0]][start_point[1]][end_point[0]][end_point[1]]
            actions.append((next_board, prob, start_point, end_point))
        return actions, prediction

def alpha_loss(predict_softmax, mcts_softmax, predict_value, mcts_value):
    '''
        value_error = (self_play_winner - winner) ** 2
        policy_error = torch.sum((-self_play_probas * (1e-6 + probas).log()), 1)
        total_error = (value_error.view(-1) + policy_error).mean()
    '''
    value_loss = F.mse_loss(predict_value.view(-1), mcts_value)
    policy_loss = -torch.mean(torch.sum(mcts_softmax*torch.log(predict_softmax), 1))
    loss = value_loss + policy_loss
    # backward and optimize
    return value_loss, policy_loss, loss

if __name__ == "__main__":
    # model = Extractor()

    # input = torch.randn(64, INPLANE, BOARD_WIDTH, BOARD_HEIGHT)

    # output = model(input)

    # print(output.shape)

    # policy_model = PolicyNet(OUTPLANES_MAP, OUTPLANES)

    # print(policy_model(output).shape)

    # value_net = ValueNet(OUTPLANES_MAP, OUTPLANES)

    # print(value_net(output).shape)
    from utils.game import Game
    game = Game()
    model = DualResNet()
    probability, prediction = model.policyvalue_function(game)
    print(probability, prediction)