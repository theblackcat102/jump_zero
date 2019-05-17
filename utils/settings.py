

BOARD_WIDTH = 8
BOARD_HEIGHT = 8

C_PUCT = 4.0

HISTORY_RECORDS = 3

BLOCKS = 3 # number of residual block

INPLANE = (HISTORY_RECORDS + 1)*2 + 1
OUTPLANES = BOARD_WIDTH*BOARD_HEIGHT

OUTPLANES_MAP = 128 # kernel number
# number of self play round to carry before backprop neural network
PARALLEL_SELF_PLAY = 300
# mcts playout round, longer playout lead to higher memory leak
PLAYOUT_ROUND = 100
# how many latest rounds used for training
SELF_TRAINING_ROUND = 500
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
## Activate MCTS
MCTS_FLAG = True
## Epsilon for Dirichlet noise
EPS = 0.3
## Alpha for Dirichlet noise
ALPHA = 0.03
## Batch size for evaluation during MCTS
BATCH_SIZE_EVAL = 2
## Number of self-play before training
SELF_PLAY_MATCH = PARALLEL_SELF_PLAY
## Number of moves before changing temperature to stop
## exploration for more steps for large possible steps
TEMPERATURE_MOVE = 50

# storage
IP= '192.168.0.103'

MODEL_DIR = './checkpoint'
PROCESS_TIMEOUT = 700