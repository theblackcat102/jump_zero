# jump_zero

## Description

This is a homework for 2019 NCTU Introduction to Artificial Intelligent. The game and rules were provided by the course TA.

### Some Rules

1. only 1 CPU core and 2G of RAM is available for your program

2. time limit is 5 seconds for each rounds

3. for more details about the game please refer to the [Game rules] section.

4. No external package should be used for the final program, however numpy is provided ( well I guess I can use any libraries to train my AI agent then )

PS: yeah, these are some weird rules

## We decided to implement a version of AlphaZero to play a kind of jump chess by Silver et al

For more details about alpha zero check out these articles 

0. (A general reinforcement learning algorithm thatmasters chess, shogi and Go through self-play)[https://deepmind.com/documents/260/alphazero_preprint.pdf]

1. (A Simple Alpha(Go) Zero Tutorial)[https://web.stanford.edu/~surag/posts/alphazero.html]

2. (AlphaZero实践——中国象棋（附论文翻译）)[https://zhuanlan.zhihu.com/p/34433581]

3. (Self-Play)[https://hackernoon.com/self-play-1f69ceb06a4d]

4. (Mastering the Game of Go without Human Knowledge)[https://deepmind.com/documents/119/agz_unformatted_nature.pdf]

## Game rules
In the following are the rules of game with highlighted text indicating the changes:

 Each player has 9 pieces initially, arranged as file provided before.

 Black moves first.

 The valid moves are:

 Move one piece to an adjacent unoccupied square.

 Have one piece hop over another piece of either player to an unoccupied square. Multiple hops by
one piece can be taken within one move as long as all the hops are valid and form a sequence.
However, there is a maximum hop count equal to 99. Any move that includes more than
99(>99) hops is considered as invalid move.

 When hopping over an opponent’s piece, the opponent’s piece is considered “captured” and removed from the board.

 The overall objective is to move as many of a player’s pieces as possible to the shaded squares in the opposite side (the target region).

 A player’s move is skipped if he/she makes an invalid move or has no piece remaining on the board. The other player will continue to play. Skipping will also happen when all pieces of a player don’t have any valid move or your program exceeds the time limit(5 second).

 The game ends when

 Either players have all his/her pieces on the board in the target region, or  Amaximumof200movesperplayerhasbeenplayed.

 At the end, the score of a player is the number of pieces placed in the target region. The player with the higher score wins the game.
The tournament rules are exactly the same as mentioned before.

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

To move the final model's weights to a numpy version of the same neural network, please read the code in models/convert.py first
```
    python -m models.convert
```


## Issues

1. Random memory leak from pytorch in individual process when self play in MCTS is too high ( > 200 )

## Code reference

1. https://github.com/junxiaosong/AlphaZero_Gomoku

2. https://github.com/dylandjian/SuperGo

## Fun Fact
Alpha zero algorithm seems to be patented by DeepMind : https://patentscope2.wipo.int/search/en/detail.jsf?docId=WO2018215665&fbclid=IwAR1FX90V0xsaf6FZUA3U7DEIMC_xGKA4GaCq4Fgg31hFebo7QKM0CUk3Uhw

Please don't sue me please, I do not own any of these except the rules and code