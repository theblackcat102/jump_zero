import numpy
import numpy as np
from utils.rules import *
from utils.game import init_board
from utils.settings import BOARD_HEIGHT, BOARD_WIDTH, C_PUCT, PLAYOUT_ROUND
import logging

logger = logging.getLogger('mcts')
fh = logging.FileHandler('mcts_error.log')
fh.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def random_rollout(available_steps):
    # we will not bother to normalize this, since we just select the max ones
    return np.random.rand(available_steps)

def uniform_probability_steps(available_steps):
    return np.array(np.ones(available_steps)/available_steps)

class TreeNode(object):
    '''
    https://github.com/initial-h/AlphaZero_Gomoku_MPI/blob/master/mcts_pure.py
    A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    '''
    __slots__ = ('_parent', '_children', '_n_visits', 'start', 'end','eaten', '_Q', '_u', '_P')

    def __init__(self, parent, prior_p, start=None, end=None, eaten=0):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self.start = start
        self.end = end
        self.eaten = eaten
        self._Q = 0
        self._u = 0
        self._P = prior_p # value policy output

    def expand(self, actions):
        '''
        actions : a list of (board matrix, probability )
        Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
        according to the policy function.
        '''
        for (board, prob, start, end, eaten) in actions:
            board_str = board.tostring()
            if board_str not in self._children:
                self._children[board_str] = TreeNode(parent=self, 
                    prior_p=prob,
                    start=start,
                    end=end,
                    eaten=eaten)
        # expand all children that under this state

    def select(self, c_puct):
        '''
        Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        '''
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        '''
        Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        '''
        self._n_visits += 1
        # update visit count
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
        # update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)

    def update_recursive(self, leaf_value):
        '''
        Like a call to update(), but applied recursively for all ancestors.
        '''
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
            # every step for revursive update,
            # we should change the perspective by the way of taking the negative
        self.update(leaf_value)

    def get_value(self, c_puct=C_PUCT):
        '''
        Calculate and return the value for this node.
        It is a combination of leaf evaluations Q,
        and this node's prior adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
        value Q, and prior probability P, on this node's score.
        '''
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        '''
        check if it's leaf node (i.e. no nodes below this have been expanded).
        '''
        return len(self._children) == 0

    def is_root(self):
        return self._parent is None


class MCTS:

    def __init__(self, policy_value_fn, c_puct=C_PUCT, n_playout=PLAYOUT_ROUND, self_play=True):
        self._root = TreeNode(parent=None, prior_p=1.0, start=(0, 0), end=(0, 0) )
        self._c_puct = c_puct
        self._policy = policy_value_fn # output list of (move, prob), and envaluation value
        self._n_playout = n_playout
        self._self_play = self_play

    def _playout(self, game):
        '''
        simulation, random choose possible step between two player until a winner is found
        state is a game instance
        ref : https://medium.com/@quasimik/implementing-monte-carlo-tree-search-in-node-js-5f07595104df
        '''
        # 1. selection
        node = self._root
        # traverse until the leaf node
        end = False
        reward, depth = 0, 0
        while True:
            if node.is_leaf() or depth > 400:
                break
            # Greedily select next move.
            node_key, node = node.select(self._c_puct)
            # update state
            board = np.fromstring(node_key, dtype=int).reshape(BOARD_WIDTH, BOARD_HEIGHT)
            end, _, reward = game.update_state(board)
            depth += 1

        if end:
            value = reward
        else:
            probability, prediction = self._policy(game)
            node.expand(probability)
            value = prediction
            # value *= (node.eaten+1)
        del game
        # backpropagation
        node.update_recursive(-value) # why negative ?

    def get_move_visits(self, state, temperature=1e-3):
        '''
        Run all playouts sequentially and return the available actions and
        their corresponding visiting times.
        state: the current game instance
        '''
        current_color = state.current
        # number of simulation to run
        for _ in range(self._n_playout):
            self._playout(state.copy())

        if len(self._root._children) == 0:
            logger.warning('Step: {}, current player: {}\nBoard: \n{}'.format(state.player_step+state.opponent_step , state.current, state.board ))
            raise ValueError('no steps found')

        visits = []
        for node_key, node in self._root._children.items():
            visits.append((np.fromstring(node_key, dtype=int).reshape(BOARD_WIDTH, BOARD_HEIGHT), node._n_visits, node.start, node.end))
        acts, visits, starts, ends = zip(*visits)

        # convert visits 
        visits = list(visits)
        for idx in range(len(visits)):
            y_diff = starts[idx][1] - ends[idx][1]
            if y_diff < 0 and current_color == 1:
                visits[idx] += 1
            if y_diff > 0 and current_color == -1:
                visits[idx] += 1

        act_probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))
        # we will use save this for later
        return acts, act_probs, generate_mcts_softmax(act_probs, starts, ends).flatten()

    def get_action(self, game, temp=1e-3, return_prob=0):
        acts, probability, _ = self.get_move_visits(game, temperature=temp)
        # pick a random move
        valid_move_count = len(probability)
        if self._self_play:
            move_idx = np.random.choice(valid_move_count, 
                p=(1-EPS)*probability + EPS*np.random.dirichlet(ALPHA*np.ones(valid_move_count))
            )
        else:
            move_idx = np.random.choice(valid_move_count, p=probability)
        step = acts[move_idx]
        return step


    def update_with_move(self, new_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if isinstance(new_move, int):
            new_move = str(new_move)
        elif isinstance(new_move, str) is False:
            new_move = new_move.tostring()

        if new_move in self._root._children:
            self._root = self._root._children[new_move]
            self._root._parent = None
        else:
            # maybe act as reset?
            self._root = TreeNode(None, 1.0)

if __name__ == "__main__":
    from models.dualresnet import DualResNet
    from utils.game import Game


        
