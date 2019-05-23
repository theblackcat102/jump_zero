from rules import *
import numpy as np

init_board = np.array([[ 1, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 1, 0, 0, 0, 0, 0, 2],
                        [ 1, 0, 1, 0, 0, 0, 2, 0],
                        [ 0, 1, 0, 0, 0, 2, 0, 2],
                        [ 1, 0, 1, 0, 0, 0, 2, 0],
                        [ 0, 1, 0, 0, 0, 2, 0, 2],
                        [ 1, 0, 0, 0, 0, 0, 2, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 2]])


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
    
    def copy(self):
        game = Game(self.player)
        game.board = np.copy(self.board)
        game.player = self.player
        game.opponent = 1 if self.player == 2 else 2
        # 1(black) always start first
        game.current = self.current
        # game.history = self.history
        game.player_step = self.player_step
        game.opponent_step = self.opponent_step
        # game.availables = next_steps(game.board, self.player)
        return game
