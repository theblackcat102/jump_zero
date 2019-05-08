import numpy as np
from utils.rules import *
from utils.settings import *

init_board = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0], 
                [ 0, 1, 0, 0, 0, 0, 0,-1],
                [ 1, 0, 1, 0, 0, 0,-1, 0],
                [ 0, 1, 0, 0, 0,-1, 0,-1],
                [ 1, 0, 1, 0, 0, 0,-1, 0],
                [ 0, 1, 0, 0, 0,-1, 0,-1],
                [ 1, 0, 0, 0, 0, 0,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 0,-1], ])

class Game:

    def __init__(self, player=1):
        # initial board placement
        self.board = init_board
        self.player = player
        self.opponent = 1 if player == -1 else -1
        # 1(black) always start first
        self.current = 1
        self.history = []
        self.player_step = 0
        self.availables = next_steps(self.board, player)
        self.opponent_step = 0

    def update_state(self, board):
        self.history.append(self.board)
        self.board = np.copy(board)
        check_state = has_won(self.board, self.player_step, self.opponent_step)
        self.player_step += 1
        self.opponent_step += 1
        self.current = 1 if self.current == -1 else -1
        end = False
        winner = None
        reward = -1
        if check_state != 0:
            end = True
            winner = check_state # -1 is white, 1 is black
            reward = self.reward_function(winner)
        return end, winner, reward
    
    def available_move(self, player): # player chess value
        '''
            Output result for feature extractor
        '''
        possible_steps = next_steps(self.board, player)
        return [generate_extractor_input(step, self.history+[self.board], self.current) for step in possible_steps ]

    def reward_function(self, winner):
        if winner == self.player:
            return 1
        elif winner == 2: # draw
            return 0.0
        return -1

    def do_move(self, new_board):
        return self.update_state(new_board)

    def current_player(self):
        return self.current

    def generate_nn_input(self):
        inputs = np.zeros((INPLANE, BOARD_WIDTH, BOARD_HEIGHT, ))
        if self.current_player() == 1:
            inputs[-1, :, :] = np.ones((BOARD_WIDTH, BOARD_HEIGHT)).astype('float')
        inputs[0, :, :] = extract_chess(self.board, 1)
        inputs[HISTORY_RECORDS, :, :] = extract_chess(self.board, -1)
        for idx in range(min(HISTORY_RECORDS-1, len(self.history))):
            inputs[idx+1, BOARD_WIDTH, :, :] = extract_chess(self.history[-1*idx], 1)
            inputs[idx+HISTORY_RECORDS+1, :, : ] = extract_chess(self.history[-1*idx], -1)
        return inputs

    def current_state(self):
        return self.generate_nn_input()

if __name__ == "__main__":
    game = Game()
    output = game.generate_nn_input()
    print(output.shape)
    print(output[:, :, -1, :])