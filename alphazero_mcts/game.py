from rules import *
import numpy as np
from copy import deepcopy

init_board = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 1, 0, 0, 0, 0, 0, 2],
                        [ 1, 0, 1, 0, 0, 0, 2, 0],
                        [ 0, 1, 0, 0, 0, 2, 0, 2],
                        [ 1, 0, 1, 0, 0, 0, 2, 0],
                        [ 0, 1, 0, 0, 0, 2, 0, 2],
                        [ 1, 0, 0, 0, 0, 0, 2, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 2]])
HISTORY_RECORDS = 3
STORE_HISTORY = (HISTORY_RECORDS+1)*2

BOARD_WIDTH = 8
BOARD_HEIGHT = 8

HISTORY_RECORDS = 3

BLOCKS = 5 # number of residual block

INPLANE = (HISTORY_RECORDS + 1)*2 + 1
OUTPLANES = BOARD_WIDTH*BOARD_HEIGHT

OUTPLANES_MAP = 64 # kernel number
# number of self play round to carry before backprop neural network
PARALLEL_SELF_PLAY = 20
# mcts playout round, longer playout lead to higher memory leak
PLAYOUT_ROUND = 120
# how many latest rounds used for training
SELF_TRAINING_ROUND = 60
# fixed learning rate
LR = 0.01
## Number of MCTS simulation
MCTS_SIM = 64
## Exploration constant
C_PUCT = 0.2
## L2 Regularization
L2_REG = 0.0001
## Momentum
MOMENTUM = 0.9

class Game:

    def __init__(self, player=1):
        # initial board placement
        self.board = init_board
        self.player = player
        self.opponent = 1 if player == 2 else 2
        # 1(black) always start first
        self.current = player
        self.history = []
        self.player_step = 0
        self.opponent_step = 0

    def update_state(self, board):
        self.history.append(self.board)
        self.history = self.history[-STORE_HISTORY:]

        self.board = np.copy(board).astype('int')

        if self.current == self.player:
            self.player_step += 1
        else:
            self.opponent_step += 1

        check_state = has_won(self.board, self.opponent_step, self.player_step)
        # switch side
        self.current = 1 if self.current == 2 else 2
        end = False
        winner = check_state
        reward = 0
        if check_state != 0: # tie=3, win=1, lose=-2
            end = True
            winner = check_state # -1 is white, 1 is black
            reward = self.reward_function(winner)
        return end, winner, reward

    def legal_move(self):
        return next_steps(self.board.copy().astype('int'), self.current)

    def reward_function(self, winner):
        if winner == self.player:
            return 1
        elif winner == 3: # draw
            return 0.0
        return -1

    def do_move(self, new_board):
        return self.update_state(new_board)

    def current_player(self):
        return self.current

    def generate_nn_input(self):
        inputs = np.zeros((INPLANE, BOARD_WIDTH, BOARD_HEIGHT, ))
        inputs[0, :, :] = extract_chess(self.board, 1)
        inputs[HISTORY_RECORDS, :, :] = extract_chess(self.board, -1)
        for idx in range(min(HISTORY_RECORDS-1, len(self.history))):
            inputs[idx+1, :, :] = extract_chess(self.history[-1*idx], 1)
            inputs[idx+HISTORY_RECORDS+1, :, : ] = extract_chess(self.history[-1*idx], -1)

        if self.current == 1:
            inputs[-1, :, :] = np.ones((BOARD_WIDTH, BOARD_HEIGHT)).astype('float')
        else:
            inputs[-1, :, :] = np.zeros((BOARD_WIDTH, BOARD_HEIGHT)).astype('float')

        return inputs

    def current_state(self):
        return self.generate_nn_input()

    def copy(self):
        return deepcopy(self)
