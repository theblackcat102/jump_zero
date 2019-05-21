# Observation Report

1. Play N batch of games and sample the latest M games for training

    The larger the N is, the noiser the training graph is. relative KL divergence seems diverge


2. Balanced win, lose, tie rounds

    A huge number of tie rounds occur in the final result. Would it be possible to ignore tie result during the training phase? As pointed out in this issue [https://github.com/leela-zero/leela-zero/issues/1656]
    "
    ...demonstrated that the no-resign games contribute a lot of moves to the training data that apparently contain very little actual information (value)...
    "

3. one round followed by one training phase

    Currently in the workings



