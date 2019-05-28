import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import numpy as np
from layers import Conv2d, BatchNorm, ReLU, Dense, Softmax, Sigmoid, Layers, Tanh
'''
Numpy version of Residual network for AlphaZero Jump
'''
BOARD_WIDTH = 8
BOARD_HEIGHT = 8

class BasicBlock():
    """
    Basic residual block with 2 convolutions and a skip connection before the last
    ReLU activation.
    """ 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # super(BasicBlock, self).__init__()
        
        self.conv1 = Conv2d(inplanes,  planes,  strides=(stride,stride) , kernel_size=(3,3),
                        padding=1, bias=False)
        self.bn1 = BatchNorm(planes)

        self.conv2 = Conv2d(planes, planes,  strides=(stride,stride) , kernel_size=(3,3),
                        padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu = ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1.forward(x)
        # print(out.flatten()[0])
        out = self.relu.forward(self.bn1.forward(out))


        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        out += residual
        out = self.relu.forward(out)

        return out

class ValueNet():

    """
    This network is used to predict which player is more likely to win given the input 'state'
    described in the Feature Extractor model.
    The output is a continuous variable, between -1 and 1. 
    """

    def __init__(self, inplanes, outplanes):
        # super(ValueNet, self).__init__()
        self.outplanes = outplanes
        self.conv = Conv2d(inplanes, 1, kernel_size=(1,1), strides=(1,1), padding=0)
        self.bn = BatchNorm(1)
        self.fc1 = Dense(outplanes, 256)
        self.fc2 = Dense(256, 1)
        self.relu = ReLU()
        self.tanh = Tanh()
        

    def forward(self, x):
        """
        x : feature maps extracted from the state
        winning : probability of the current agent winning the game
                  considering the actual state of the board
        """
 
        x = self.relu.forward(self.bn.forward(self.conv.forward(x)))
        x = x.reshape(-1, self.outplanes )
        x = self.relu.forward(self.fc1.forward(x))
        winning = self.tanh.forward(self.fc2.forward(x))
        return winning


class PolicyNet():
    def __init__(self, inplanes, outplanes):
        self.outplanes = outplanes
        self.conv = Conv2d(inplanes ,1, kernel_size=(1, 1), strides=(1,1), padding=0)
        self.bn = BatchNorm(1)
        self.softmax = Softmax()
        self.fc = Dense(outplanes, outplanes*outplanes)
        self.relu = ReLU()

    def forward(self, x):
        """
        x : feature maps extracted from the state
        probas : a NxN + 1 matrix where N is the board size
                 Each value in this matrix represent the likelihood
                 of winning by playing this intersection
        """
        x = self.relu.forward(self.bn.forward(self.conv.forward(x)))
        x = x.reshape(-1, self.outplanes )
        x = self.fc.forward(x)
        probas = self.softmax.forward(x)

        return probas

class DualResNetNumpy():
    '''
        A numpy version of ResNet since 
        this course is too shit to use external library
    '''
    def __init__(self, inplanes=9, res_block=4, conv_channel=64):
        # feature extraction
        self.conv1 = Conv2d(inplanes, conv_channel, strides=(1,1) , kernel_size=(3,3),
                        padding=1, bias=False)
        self.bn1 = BatchNorm(conv_channel)
        self.residual_block = []
        for idx in range(res_block):
            self.residual_block.append( BasicBlock(conv_channel, conv_channel) )
        
        # policy block
        self.policy = PolicyNet(conv_channel, 8*8)

        self.value = ValueNet(conv_channel, 8*8)


    def forward(self, inputs):
        x = self.conv1.forward(inputs)  
        x = self.bn1.forward(x)

        for idx, res_block in enumerate(self.residual_block):
            # print('block {}'.format(idx))
            x = res_block.forward(x)
        feature = x
        softmax_output = self.policy.forward(feature)
        value = self.value.forward(feature)
        return value, softmax_output

    def policyvalue_function(self, game):
        feature_input = np.array([game.current_state()])


        prediction, probability = self.forward(feature_input)
        prediction = prediction[0]
        probability = probability[0].reshape(BOARD_WIDTH, BOARD_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT)
        # we need to convert probability to a board(matrix) and probability(float)
        legal_moves = game.legal_move()
        actions = []
        for (next_board, start_point, end_point, eaten ) in legal_moves:
            prob = probability[start_point[0]][start_point[1]][end_point[0]][end_point[1]]
            actions.append((next_board, prob, start_point, end_point, eaten))
        return actions, prediction


    @staticmethod
    def load_state_from_pytorch(state_dict, conv_channel=64):
        # state_dict : ordered dictionary
        # print(state_dict)
        residual_block_cnt = 0
        for name, weight in state_dict.items():
            keys = name.split('.')
            level = keys[0]
            module = keys[1] 
            layer_type = keys[-2]
            param_type = keys[-1]
            if level == 'extractor':
                if module != layer_type:
                    module_int = int(module[-1])
                    if layer_type == 'conv2':
                        if param_type == 'weight':
                            residual_block_cnt += 1

        print('residual blocks : {}, channel size {}'.format(residual_block_cnt, conv_channel))
        model = DualResNetNumpy(res_block=residual_block_cnt, conv_channel=conv_channel)
        state = {}
        for name, weight in state_dict.items():
            keys = name.split('.')
            level = keys[0]
            module = keys[1]
            layer_type = keys[-2]
            param_type = keys[-1]
            # print(name)
            weight = weight.data.numpy().astype('float32')
            if level == 'extractor':
                if module != layer_type:
                    module_int = int(module[-1])
                    if layer_type == 'conv2':
                        if param_type == 'weight':
                            # print(name)
                            model.residual_block[module_int].conv2.weights = weight.reshape(model.residual_block[module_int].conv2.shape)
                    elif layer_type == 'conv1':
                        if param_type == 'weight':
                            # print(name)
                            model.residual_block[module_int].conv1.weights = weight.reshape(model.residual_block[module_int].conv1.shape)                        
                    elif layer_type == 'bn1':
                        if param_type == 'weight':
                            # print(name)
                            model.residual_block[module_int].bn1.weight = weight.reshape(model.residual_block[module_int].bn1.channel_size)
                        elif param_type == 'bias':
                            # print(name)
                            model.residual_block[module_int].bn1.bias = weight.reshape(model.residual_block[module_int].bn1.channel_size)
                        elif param_type == 'running_var':
                            # print(name)
                            model.residual_block[module_int].bn1.running_var = weight.reshape(model.residual_block[module_int].bn1.channel_size)
                        elif param_type == 'running_mean':
                            # print(name)
                            model.residual_block[module_int].bn1.running_mean = weight.reshape(model.residual_block[module_int].bn1.channel_size)
                    elif layer_type == 'bn2':
                        if param_type == 'weight':
                            # print(name)
                            model.residual_block[module_int].bn2.weight = weight.reshape(model.residual_block[module_int].bn2.channel_size)
                        elif param_type == 'bias':
                            # print(name)
                            model.residual_block[module_int].bn2.bias = weight.reshape(model.residual_block[module_int].bn2.channel_size)
                        elif param_type == 'running_var':
                            # print(name)
                            model.residual_block[module_int].bn2.running_var = weight.reshape(model.residual_block[module_int].bn2.channel_size)
                        elif param_type == 'running_mean':
                            # print(name)
                            model.residual_block[module_int].bn2.running_mean = weight.reshape(model.residual_block[module_int].bn2.channel_size)
                else: # first conv and batchnorm
                    if layer_type == 'conv1':
                        # print(name)
                        model.conv1.weights = weight.reshape(model.conv1.shape)
                    elif layer_type == 'bn1':
                        if param_type == 'weight':
                            # print(name)
                            model.bn1.weight = weight.reshape(model.bn1.channel_size)
                        elif param_type == 'bias':
                            # print(name)
                            model.bn1.bias = weight.reshape(model.bn1.channel_size)
                        elif param_type == 'running_var':
                            model.bn1.running_var = weight.reshape(model.bn1.channel_size)
                        elif param_type == 'running_mean':
                            # print(name)
                            model.bn1.running_mean = weight.reshape(model.bn1.channel_size)
            elif level == 'policy_model':
                if layer_type == 'conv':
                    if param_type == 'weight':
                        # print(name)
                        model.policy.conv.weights = weight.reshape(model.policy.conv.shape)
                    elif param_type == 'bias':
                        # print(name)
                        model.policy.conv.bias = weight
                elif layer_type == 'bn':
                    if param_type == 'weight':
                        # print(name)
                        model.policy.bn.weight = weight.reshape(model.policy.bn.channel_size)
                    elif param_type == 'bias':
                        # print(name)
                        model.policy.bn.bias = weight.reshape(model.policy.bn.channel_size)
                    elif param_type == 'running_var':
                        # print(name)
                        model.policy.bn.running_var = weight.reshape(model.policy.bn.channel_size)
                    elif param_type == 'running_mean':
                        # print(name)
                        model.policy.bn.running_mean = weight.reshape(model.policy.bn.channel_size)
                elif layer_type == 'fc':
                    if param_type == 'weight':
                        # print(name)
                        model.policy.fc.weight = weight
                    elif param_type == 'bias':
                        # print(name)
                        model.policy.fc.bias = weight
            elif level == 'value_model':
                if layer_type == 'conv':
                    if param_type == 'weight':
                        # print(name)
                        model.value.conv.weights = weight.reshape(model.value.conv.shape)
                    elif param_type == 'bias':
                        # print(name)
                        model.value.conv.bias = weight
                elif layer_type == 'bn':
                    if param_type == 'weight':
                        # print(name)
                        model.value.bn.weight = weight.reshape(model.value.bn.channel_size)
                    elif param_type == 'bias':
                        # print(name)
                        model.value.bn.bias = weight.reshape(model.value.bn.channel_size)
                    elif param_type == 'running_var':
                        # print(name)
                        model.value.bn.running_var = weight.reshape(model.value.bn.channel_size)
                    elif param_type == 'running_mean':
                        # print(name)
                        model.value.bn.running_mean = weight.reshape(model.value.bn.channel_size)
                elif layer_type == 'fc1':
                    if param_type == 'weight':
                        # print(name)
                        model.value.fc1.weight = weight
                    elif param_type == 'bias':
                        # print(name)
                        model.value.fc1.bias = weight
                elif layer_type == 'fc2':
                    if param_type == 'weight':
                        # print(name)
                        model.value.fc2.weight = weight
                    elif param_type == 'bias':
                        # print(name)
                        model.value.fc2.bias = weight
        return model

if __name__ == "__main__":
    import pickle
    # import torch, os
    from tqdm import tqdm
    from game import Game

    game = Game()
    
    # from models.dualresnet import DualResNet
    # MODEL_DIR = './checkpointv2'
    # checkpoint = torch.load(os.path.join(MODEL_DIR, 'DualResNetv6.2_77.pt'), map_location='cpu')
    # model = DualResNet()
    # model.load_state_dict(checkpoint['network'])
    # model.eval()
    # block = DualResNetNumpy.load_state_from_pytorch(checkpoint['network'])
    with open('numpy_nn.pkl', 'rb') as f:
        block = pickle.load(f)
    # block = DualResNetNumpy(inplanes=9, res_block=5, conv_channel=64)
    prob, value = block.policyvalue_function(game)
    # print(prob, value)
    # avg_val = 0
    # for i in tqdm(range(1000), total=1000):
    inputs = np.random.randint(0,2, size=(100, 9, 8, 8))
    for i in range(100):
        inputs[i, -1, :, :] = np.random.choice([1.0, 0.0])
    value, policy = block.forward(inputs)
    py_policy, py_value = model(torch.from_numpy(inputs).type('torch.FloatTensor'))
    py_policy = np.exp(py_policy.data.cpu().numpy())
    py_value = py_value.data.cpu().numpy()

    print('value MSE {}'.format(np.mean( ( py_value - value )**2 )))
    print('softmax MSE {}'.format(np.mean( ( py_policy - policy )**2 )))
