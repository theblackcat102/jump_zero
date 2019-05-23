import numpy as np

BOARD_WIDTH = 8
BOARD_HEIGHT = 8


def check_boundary(points):
    x, y = points
    if 0 <= x and x < BOARD_HEIGHT and 0 <= y and y < BOARD_WIDTH:
        return True
    return False


def point_within_boundary(origin, points, board, move):
    if check_boundary(points) and board[points[0]][points[1]] == 0:
        # move points
        new_board = np.copy(board)
        new_board[origin[0], origin[1]] = 0
        new_board[points[0]][points[1]] = move
        return [(new_board, origin, points, 0)]
    return []


def extract_chess(board, piece_number):
    # extract for individual pieces for neural network state snapshot
    result = np.zeros((BOARD_WIDTH, BOARD_HEIGHT)).astype('float')
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if board[x][y] == piece_number:
                result[x][y] = 1.0
    return result


def has_won(board, white_step, black_step):
    '''
    1 is black, 2 is white, 0 is empty
    @params: 
    white_step: how many white steps has taken, int
    black_step: how many black steps has taken, int
    board: 8x8 matrix
    '''

    board = board.astype('int')
    unique, counts = np.unique(board.flatten(), return_counts=True)
    result = dict(zip(unique, counts))

    black_region = board[:, :2] 

    white_cnt_blk_region, black_cnt_blk_region = 0, 0
    unique, counts = np.unique(black_region, return_counts=True)
    black_region_result = dict(zip(unique, counts))
    if 2 in black_region_result:
        white_cnt_blk_region = black_region_result[2]

    white_region = board[:, -2:]
    unique, counts = np.unique(white_region, return_counts=True)
    white_region_result = dict(zip(unique, counts))
    if 1 in white_region_result:
        black_cnt_blk_region = white_region_result[1]

    if len(result) == 2: # only one side chess left
        if white_cnt_blk_region > 0:
            return 2
        elif black_cnt_blk_region > 0:
            return 1
        return 3 # tie

    if (white_step + black_step) >= 400:
        if white_cnt_blk_region == black_cnt_blk_region:
            return 3 # tie
        return 2 if white_cnt_blk_region > black_cnt_blk_region else 1

    if white_cnt_blk_region+black_cnt_blk_region == 0:
        return 0
    # End Game 1: Either player has all his/her pieces on the board in the target region
    # End game 2: A maximum of 200 moves per player has been played
    if (2 in result and result[2] == white_cnt_blk_region) or \
        ( 1 in result and result[1] == black_cnt_blk_region) or \
        white_step >= 200 or black_step >= 200:
        if white_cnt_blk_region == black_cnt_blk_region:
            return 3 # tie
        return 2 if white_cnt_blk_region > black_cnt_blk_region else 1

    # not end yet
    return 0 # continue


def dfs(board, start_pt, end_pt, check_pt, current_eaten, target_eaten, player, visited, hop_depth):
    if not check_boundary(end_pt):
        return [], visited
    if visited.get(end_pt):
        return [], visited
    if board[check_pt[0], check_pt[1]] == 0:
        return [], visited
    if hop_depth >= 99:
        return [], visited
    opponent = 2 if player == 1 else 1
    if board[check_pt[0], check_pt[1]] == opponent:
        current_eaten += 1
    if current_eaten == target_eaten and start_pt == end_pt:
        return [list(end_pt)], visited

    visited[end_pt] = True
    hop_depth += 1
    for offsets in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        tmp, visited = dfs(board, start_pt, (end_pt[0] + (offsets[0]*2), end_pt[1] + (offsets[1]*2)),
                           (end_pt[0] + (offsets[0]), end_pt[1] + (offsets[1])), current_eaten,
                           target_eaten, player, visited, hop_depth + 1)
        if tmp:
            return [list(end_pt)] + tmp, visited
    # no path found
    visited.pop(end_pt)
    return [], visited


def get_all_move(board, start_pt, end_pt, eaten, player):
    ans, _ = dfs(board, start_pt, end_pt, start_pt, 0, eaten, player, {}, 0)
    ans.reverse()
    return ans


def hop_board(origin, condition_points, points_to_move, board, move, origin_start, eaten, hop_depth=0, direction='L', visited={}):
    # check if boundary satisfied
    if check_boundary(points_to_move) is False:
        return []
    if board[points_to_move[0], points_to_move[1]] != 0 and direction != 'N':
        return []
    if visited.get(points_to_move) is not None and board[condition_points[0], condition_points[1]] == move:
        return []
    # check hop limit
    if hop_depth >= 99:
        return []
    
    # check if condition points satisfied:
    if np.abs(board[condition_points[0], condition_points[1]]) <= 0.5:
        return []
    ans = []
    opposite = 1 if move == 2 else 2

    board[origin[0], origin[1]] = 0
    board[points_to_move[0], points_to_move[1]] = move
    if board[condition_points[0], condition_points[1]] == opposite:
        board[condition_points[0], condition_points[1]] = 0
        eaten += 1
    if direction != 'N':
        ans.extend([(board, origin_start, points_to_move, eaten)])
    visited[points_to_move] = True
    for (offset, direction) in [ ((-1, 0), 'L'), ((1, 0), 'R'), ((0, -1), 'D'), ((0, 1), 'U') ]:
        ans.extend(hop_board(points_to_move, (points_to_move[0] + (offset[0]), points_to_move[1] + (offset[1])),
                             (points_to_move[0] + (offset[0]*2), points_to_move[1] + (offset[1]*2)), board.copy(),
                             move, origin_start, eaten, hop_depth+1, direction, visited))

    return ans


def next_steps(board, move=1):
    '''
        params: move =1 or 2
    '''
    possible_step = []
    if board.shape != (BOARD_WIDTH, BOARD_HEIGHT):
        raise ValueError('Board size is invalid, found shape '+ str(board.shape))
    for x in range(BOARD_WIDTH):
        for y in range(BOARD_HEIGHT):
            if int(board[x][y]) == move:
                possible_step.extend(hop_board((x, y), (x, y), (x, y), board.copy(), move, (x, y), 0, 1, 'N', {}))
                # check basic four step (R, L, U, D)
                possible_step.extend(point_within_boundary((x, y), (x + 1, y), board.copy(), move))  # right
                possible_step.extend(point_within_boundary((x, y), (x - 1, y), board.copy(), move))  # left
                possible_step.extend(point_within_boundary((x, y), (x, y + 1), board.copy(), move))  # up
                possible_step.extend(point_within_boundary((x, y), (x, y - 1), board.copy(), move))  # down
    return possible_step  # list of  8x8 matrix


if __name__ == "__main__":
    test = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 2, 0],
                     [0, 1, 0, 0, 0, 2, 0, 2],
                     [1, 0, 1, 0, 0, 1, 2, 0],
                     [0, 1, 0, 0, 0, 2, 2, 2],
                     [1, 0, 0, 0, 0, 0, 2, 0],
                     [0, 0, 0, 0, 0, 0, 0, 2], ])
    #print(get_all_move(test, (4, 5), (4, 5), 1, 1))
