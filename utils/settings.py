

BOARD_WIDTH = 8
BOARD_HEIGHT = 8

C_PUCT = 4.0

HISTORY_RECORDS = 4

BLOCKS = 2 # number of residual block

INPLANE = (HISTORY_RECORDS + 1)*2 + 1
OUTPLANES = BOARD_WIDTH*BOARD_HEIGHT
OUTPLANES_MAP = 10
PARALLEL_SELF_PLAY = 5000
PLAYOUT_ROUND = 200
DEVICE = 'cpu'

LR = 0.01
## Number of MCTS simulation
MCTS_SIM = 64
## Exploration constant
C_PUCT = 0.2
## L2 Regularization
L2_REG = 0.0001
## Momentum
MOMENTUM = 0.9
## Activate MCTS
MCTS_FLAG = True
## Epsilon for Dirichlet noise
EPS = 0.25
## Alpha for Dirichlet noise
ALPHA = 0.03
## Batch size for evaluation during MCTS
BATCH_SIZE_EVAL = 2
## Number of self-play before training
SELF_PLAY_MATCH = PARALLEL_SELF_PLAY
## Number of moves before changing temperature to stop
## exploration
TEMPERATURE_MOVE = 5 

# storage
IP= '192.168.0.103'