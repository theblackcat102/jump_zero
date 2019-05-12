from numba import jit
import numpy as np
from utils.settings import BOARD_WIDTH, BOARD_HEIGHT, HISTORY_RECORDS

@jit
def check_boundary(points):
    x, y = points
    if 0 <= x < BOARD_HEIGHT and 0 <= y < BOARD_WIDTH:
        return True
    return False


@jit
def point_within_boundary(origin, points, board, move):
    if check_boundary(points) and board[points[0]][points[1]] == 0:
        # move points
        new_board = np.copy(board)
        new_board[origin[0], origin[1]] = 0
        new_board[points[0]][points[1]] = move
        return [(new_board, origin, points)]
    return []

@jit
def extract_chess(board, piece_number):
    # extract for individual pieces for neural network state snapshot
    result = np.zeros((BOARD_WIDTH, BOARD_HEIGHT)).astype('float')
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[x][y] == piece_number:
                result[x][y] = 1.0
    return result

@jit
def has_won(board, white_step, black_step):
    '''
    1 is black, -1 is white, 0 is empty
    @params: 
    white_step: how many white steps has taken, int
    black_step: how many black steps has taken, int
    board: 8x8 matrix
    '''

    board = board.astype('int')
    unique, counts = np.unique(board.flatten(), return_counts=True)
    result = dict(zip(unique, counts))

    black_region = board[:, :2] 
    white_region = board[:, -2:]
    # print(black_region)
    # print(white_region)
    white_cnt_blk_region, black_cnt_blk_region = 0, 0

    for e in black_region.flatten():
        if e == -1:
            white_cnt_blk_region += 1

    for e in white_region.flatten():
        if e == 1:
            black_cnt_blk_region += 1

    if white_cnt_blk_region+ black_cnt_blk_region == 0:
        return 0
    # End Game 1: Either player has all his/her pieces on the board in the target region
    # End game 2: A maximum of 200 moves per player has been played
    if (-1 in result and result[-1] == white_cnt_blk_region) or \
        ( 1 in result and result[1] == black_cnt_blk_region) or \
        white_step >= 200 or black_step >= 200:

        if white_cnt_blk_region == black_cnt_blk_region:
            return 2

        return -1 if white_cnt_blk_region > black_cnt_blk_region else 1
    
    # not end yet
    return 0 # continue

def is_move_valid(board, x, y):
    if board[x][y] == 0:
        return True

@jit
def hop_board(origin, condition_points, points_to_move, board, move, hop_depth, direction, visited):
    if visited.get(points_to_move):
        return []
    # check hop limit
    if hop_depth == 99:
        return []
    # check if boundary satisfied
    if not check_boundary(condition_points) or not check_boundary(points_to_move):
        return []
    # check if condition points satisfied:
    if board[condition_points[0], condition_points[1]] != -1 and board[condition_points[0], condition_points[1]] != 1:
        return []

    ans = []
    opposite = 1 if move == -1 else - 1

    new_board = np.copy(board)
    new_board[origin[0], origin[1]] = 0
    new_board[points_to_move[0], points_to_move[1]] = move
    if new_board[condition_points[0], condition_points[1]] == opposite:
        new_board[condition_points[0], condition_points[1]] = 0
    if direction != 'N':
        ans.extend([(new_board, origin, points_to_move)])
    visited[condition_points] = True

    ans.extend(hop_board(points_to_move, (points_to_move[0] - 1, points_to_move[1]),
                         (points_to_move[0] - 2, points_to_move[1]), new_board, move, hop_depth + 1, 'L', visited))
    ans.extend(hop_board(points_to_move, (points_to_move[0] + 1, points_to_move[1]),
                         (points_to_move[0] + 2, points_to_move[1]), new_board, move, hop_depth + 1, 'R', visited))
    ans.extend(hop_board(points_to_move, (points_to_move[0], points_to_move[1] - 1),
                         (points_to_move[0], points_to_move[1] - 2), new_board, move, hop_depth + 1, 'D', visited))
    ans.extend(hop_board(points_to_move, (points_to_move[0], points_to_move[1] + 1),
                         (points_to_move[0], points_to_move[1] + 2), new_board, move, hop_depth + 1, 'U', visited))

    return ans

@jit
def next_steps(board, move=1):
    '''
        params: move =1 or -1
    '''
    possible_step = []
    if board.shape != (BOARD_WIDTH, BOARD_HEIGHT):
        raise ValueError('Board size is invalid, found shape '+ str(board.shape))
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if int(board[x][y]) == move:
                # check hop step (R, L, U, D)
                possible_step.extend(hop_board((x, y), (x, y), (x, y), board, move, 1, 'N', {}))
                # check basic four step (R, L, U, D)
                possible_step.extend(point_within_boundary((x, y), (x + 1, y), board, move))  # right
                possible_step.extend(point_within_boundary((x, y), (x - 1, y), board, move))  # left
                possible_step.extend(point_within_boundary((x, y), (x, y + 1), board, move))  # up
                possible_step.extend(point_within_boundary((x, y), (x, y - 1), board, move))  # down
    return possible_step  # list of  8x8 matrix

@jit
def generate_extractor_input(current_board, board_history, current_player ):
    '''
        Generate input for neural network, irrelevant to game play
    '''
    inputs = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, (HISTORY_RECORDS+1)*2 + 1))
    opposite = -1 if current_player == 1 else 1
    if current_player == 1:
        inputs[:, :, (HISTORY_RECORDS+1)*2] = np.ones((BOARD_WIDTH, BOARD_HEIGHT))
    output = extract_chess(current_board, current_player)
    inputs[:, :, 0] = output
    inputs[:, :, HISTORY_RECORDS+1] = extract_chess(current_board, opposite)

    for idx in range(min(HISTORY_RECORDS, len(board_history))):
        inputs[:, :, idx+1] = extract_chess(board_history[-1*idx], 1)
        inputs[:, :, idx+HISTORY_RECORDS+1] = extract_chess(board_history[-1*idx], -1)
    return inputs


# @jit(nopython=True, parallel=True)
def generate_mcts_softmax(probabilities, start_points, end_points, softmax_output = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT))):
    for idx in range(len(probabilities)):
        s_x, s_y = start_points[idx]
        e_x, e_y = end_points[idx]
        softmax_output[ s_x ][s_y][e_x][e_y] = probabilities[idx]
    return softmax_output

if __name__ == "__main__":
    # test = [[ 1, 0, 0, 0, 0, 0, 0, 0], 
    #         [ 0, 1, 0, 0, 0, 0, 0,-1],
    #         [ 1, 0, 0, 0, 1, 0,-1, 0],
    #         [ 0, 1, 0, 0, 0,-1, 0,-1],
    #         [ 1, 0, 1, 0, 0, 0,-1, 0],
    #         [ 0, 0, 0, 0, 1,-1, 0,-1],
    #         [ 1, 0, 0, 0, 0, 0,-1, 0],
    #         [ 0, 0, 0, 0, 0, 0, 0,-1], ]
    # print(str(np.asarray(test)))
    # # draw, both region has zero pieces
    # print(has_won(np.asarray(test), 200, 10))
    # draw = [[-1, 0, 0, 0, 0, 0, 0, 0], 
    #         [ 0, 1, 0, 0, 0, 0, 0,-1],
    #         [ 1, 0, 0, 0, 1, 0,-1, 0],
    #         [ 0, 1, 0, 0, 0,-1, 0,-1],
    #         [ 1, 0, 1, 0, 0, 0,-1, 0],
    #         [ 0, 0, 0, 0, 1,-1, 0,-1],
    #         [ 1, 0, 0, 0, 0, 0,-1, 0],
    #         [ 0, 0, 0, 0, 0, 0, 0, 1], ]
    # print(has_won(np.asarray(draw), 200, 10))
    # # continue game
    # print(has_won(np.asarray(test), 11, 10))
    # print(has_won(np.asarray(test), 0, 0))
    # white_won =[[-1, 0, 0, 0, 0, 0, 0, 0], 
    #             [ 0, 1, 0, 0, 0, 0, 0,-1],
    #             [ 1, 0, 1, 0, 0, 0,-1, 0],
    #             [ 0, 1, 0, 0, 0,-1, 0, 1],
    #             [-1, 0, 1, 0, 0, 0,-1, 0],
    #             [-1, 1, 0, 0, 0, 0, 0,-1],
    #             [ 1, 0, 0, 0, 0, 0,-1, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0,-1], ]
    # print(has_won(np.asarray(white_won), 200, 10))
    # black_won =[[-1, 0, 0, 0, 0, 0, 0, 0], 
    #             [ 0, 1, 0, 0, 0, 0, 0, 1],
    #             [ 1, 0, 1, 0, 0, 0,-1, 0],
    #             [ 0, 1, 0, 0, 0,-1, 0, 1],
    #             [-1, 0, 1, 0, 0, 0,-1, 0],
    #             [-1, 1, 0, 0, 0, 0, 0, 1],
    #             [ 1, 0, 0, 0, 0, 0,-1, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 1], ]
    # print(has_won(np.asarray(black_won), 200, 10))
    # # all steps in white area
    # black_won =[[-1, 0, 0, 0, 0, 0, 0, 0], 
    #             [ 0, 0, 0, 0, 0, 0, 0, 1],
    #             [ 0, 0, 0, 0, 0, 0,-1, 0],
    #             [ 0, 0, 0, 0, 0,-1, 0, 1],
    #             [-1, 0, 0, 0, 0, 0,-1, 0],
    #             [-1, 0, 0, 0, 0, 0, 0, 1],
    #             [ 0, 0, 0, 0, 0, 0,-1, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 1], ]
    # print(has_won(np.asarray(black_won), 100, 10))
    test_hop =[ [ 0, 0, 0, 0, 0, 0, 0, 0], 
                [ 0, 1, 1, 0, 0, 0, 0, 0],
                [ 0, 0, -1, -1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0], ]

    # possible_steps = hop_board(test_hop, (1,1), (1,2), (1,3), move=1)
    possible_steps = next_steps(np.array(test_hop), move=-1)
    for (step, previous_points, new_points) in possible_steps:
        print(new_points) # the new location which the piece has moved to new location
        print(step)
    possible_steps = next_steps(np.array(test_hop), move=1)
    for (step, previous_points, new_points) in possible_steps:
        print(new_points) # the new location which the piece has moved to new location
        print(step-test_hop)
    # extract_chess(black_won, 1)