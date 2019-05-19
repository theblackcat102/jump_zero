# jump_zero

## This repo implements a version of AlphaZero by Silver et al

For more details about alpha zero check out these articles 

0. (A general reinforcement learning algorithm thatmasters chess, shogi and Go through self-play)[https://deepmind.com/documents/260/alphazero_preprint.pdf]

1. (A Simple Alpha(Go) Zero Tutorial)[https://web.stanford.edu/~surag/posts/alphazero.html]

2. (AlphaZero实践——中国象棋（附论文翻译）)[https://zhuanlan.zhihu.com/p/34433581]

3. (Self-Play)[https://hackernoon.com/self-play-1f69ceb06a4d]

4. (Mastering the Game of Go without Human Knowledge)[https://deepmind.com/documents/119/agz_unformatted_nature.pdf]


## Requirements

10 core CPU with more than 32G of RAM with CUDA enabled GPU. You can lower the CPU_COUNT at utils/settings.py to fit your system core count. This specification is required for fast parallel self play.

Install the required package

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Run

Start self play
```
    python -m cli.self_play
```

## Issues

1. Random memory leak from pytorch in individual process when self play in MCTS is too high ( > 200 )
