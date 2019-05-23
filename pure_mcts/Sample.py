
import STcpClient
import numpy as np
from game import Game
from rules import get_all_move
from mcts import MCTS as MCTS_Pure
'''
    輪到此程式移動棋子
    board : 棋盤狀態(list of list), board[i][j] = i row, j column 棋盤狀態(i, j 從 0 開始)
            0 = 空、1 = 黑、2 = 白
    is_black : True 表示本程式是黑子、False 表示為白子

    return step
    step : list of list, step = [(r1, c1), (r2, c2) ...]
            r1, c1 表示要移動的棋子座標 (row, column) (zero-base)
            ri, ci (i>1) 表示該棋子移動路徑
'''


def GetStep(board, is_black):
    # fill your program here
    if is_black:
        game.current = 1
    else:
        game.current = 2
    game.board = np.copy(board)
    move, start, end, eaten = mcts_player_1.get_action(game.copy())
    mcts_player_1.update_with_move(-1)
    step = [list(start), list(end)]
    if abs(start[0] - end[0]) > 1 or abs(start[1] - end[1]) > 1:
        step = get_all_move(game.board, start, end, eaten, game.current)
    return step
    pass


game = Game()
mcts_player_1 = MCTS_Pure(c_puct=5, n_playout=30)
while(True):
    (stop_program, id_package, board, is_black) = STcpClient.GetBoard()
    if(stop_program):
        break
    
    listStep = GetStep(board, is_black)
    STcpClient.SendStep(id_package, listStep)
