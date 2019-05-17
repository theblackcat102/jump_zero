import numpy as np
from models.utils import im2col_indices, col2im_indices

'''
numpy net
implementation of neural networks using numpy
https://github.com/theblackcat102/DL_2018
'''

def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

class Layers:
    def forward(self, x):
        return None

    def backward(self, grad):
        return grad

    def update(self):
        return None
    
    def update_lr(self, lr):
        pass

    def has_weight(self):
        return False


class Conv2d(Layers):
    def __init__(self, channel ,filters, kernel_size=(3,3), strides=(1,1),padding=1, l2_regularization=None, bias=True):
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("kernel size must be same")

        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.filters = filters
        self.channel_size = channel
        self.l2_regularization = l2_regularization
        # self.w_optimizer = SGD(learning_rate=0.001)
        # self.b_optimizer = SGD(learning_rate=0.001)
        self.first_pass = False
        self.use_bias = bias
        self.initial_weights(1.0)
        self.shape = (self.filters, self.channel_size, self.kernel_size[0], self.kernel_size[1])
    
    def initial_weights(self, C):
        sigma = 0.01
        # self.weights = np.random.randn(self.filters, 
        #         self.channel_size, self.kernel_size[0], 
        #         self.kernel_size[1]) * np.sqrt(2. / (self.kernel_size[1]*self.kernel_size[0]*self.channel_size*self.filters))
        self.weights = np.random.uniform(
            low=-np.sqrt(6 / (self.filters + C)),
            high=np.sqrt(6 / (self.filters + C)),
            size=(self.filters, self.channel_size, self.kernel_size[0], self.kernel_size[1])
        )
        if self.use_bias:
            self.bias = np.zeros((self.filters, 1))
        # self.padding = 1

    def set_optimizer(self, optimizer):
        self.w_optimizer = optimizer.copy()
        if self.use_bias:
            self.b_optimizer = optimizer.copy()

    def forward(self, x):
        N, C, H, W = x.shape
        self.channel_size = C
        if self.weights is None:
            self.initial_weights(C)
            self.first_pass = True
        assert len(x.shape) == 4

        self.input_size = (N, C, H, W)
        # print(self.input_size)
        padding = self.padding
        H_out = (H - self.kernel_size[0] + 2*padding ) // self.strides[0] + 1
        W_out = (W - self.kernel_size[1] + 2*padding ) // self.strides[1] + 1

        X_col = im2col_indices(x, self.kernel_size[0], self.kernel_size[1], padding=padding, stride=self.strides[1]).astype('float32')
        self.X_col = X_col

        W_col = self.weights.reshape(self.filters, -1)
        X_col_output = W_col.dot(X_col) #+ self.bias
        # print('weight', X_col_output.flatten()[0])
        if self.use_bias:
            X_col_output += self.bias
        output = X_col_output.reshape(self.filters, H_out, W_out, N)
        output = output.transpose(3, 0, 1, 2)
        return output
        
    def backward(self, grad):

        self.d_b = np.sum(grad, axis=(0, 2, 3))
        # print(self.d_b.shape)
        self.d_b = self.d_b.reshape(self.filters, -1)

        dout = grad.transpose(1,2,3,0).reshape(self.filters, -1)
        d_w = dout.dot(self.X_col.T)
        d_w = d_w.reshape(self.filters, -1)
        self.d_w = d_w
        if self.l2_regularization is not None:
            self.d_w += self.l2_regularization * ( self.weights.reshape(self.d_w.shape) )

        W_shape = self.weights.reshape(self.filters, -1)
        dX = W_shape.T.dot(dout)
        dX = col2im_indices(dX, self.input_size, self.kernel_size[0], self.kernel_size[1], padding=self.padding, stride=self.strides[1])

        return dX
    
    def update(self):
        weight_diff = self.w_optimizer.update(self.d_w.reshape(self.weights.shape))
        bias_diff = self.b_optimizer.update(self.d_b)
        # print(weight_diff)
        self.weights -= weight_diff
        self.bias -= bias_diff
    
    def update_lr(self, lr):
        self.w_optimizer.update(lr)
        self.b_optimizer.update(lr)
    
    def has_weight(self):
        return True

class Flatten(Layers):
    def __init__(self):
        self.reshaped = False

    def forward(self, x):
        if len(x.shape) > 2:
            self.reshaped = True
            N, C, H, W = x.shape
            self.input_size = x.shape
            return x.reshape(N, C*H*W)
        return x

    def backward(self, grad):
        if self.reshaped:
            N, C, H, W = self.input_size
            # print('gradient shape')
            # print(grad.shape)
            output = grad.reshape(self.input_size)
            # print(output.shape)
            return output
        return grad

class MaxPooling(Layers):
    '''
        2D max pooling
    '''
    def __init__(self, pool_size=2, strides=None, padding='valid', channel_first=True):
        '''
            Only padding valid is available, pool size must be rectangle
        '''
        self.pool_size = pool_size
        self.strides = 1 if strides is None else strides

        self.channel_first = channel_first
    

    def forward(self, x):

        if self.channel_first:
            batch_size, channel_size, f1, f2 = x.shape
            x_reshaped = x.reshape(batch_size*channel_size, 1, f1, f2)
        else:
            batch_size, f1, f2, channel_size = x.shape
            x_reshaped = np.rollaxis(x, 3, 1)
            x_reshaped = x_reshaped.reshape(batch_size*channel_size, 1, f1, f2)

        assert f1 == f2 # image size must be same


        # 2 dimension 

        X_col = im2col_indices(x_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.strides).astype('float32')

        max_idxs = np.argmax(X_col, axis=0)

        # cache 
        self.input_size = (batch_size, channel_size, f1, f2)
        self.output = X_col.shape
        self.max_idxs = max_idxs

        out = X_col[max_idxs, np.arange(max_idxs.size)]
        # h_out = int(np.sqrt(len(out) // batch_size // channel_size))
        h_out = (f1 - self.pool_size ) // self.strides + 1
        out = out.reshape(h_out, h_out, batch_size, channel_size)
        out = out.transpose(2, 3, 0, 1)
        if self.channel_first:
            return out
        else:
            return np.rollaxis(out, 1, 3)
    
    def backward(self, grad):
        batch_size, f1, f2, channel_size = self.input_size
        dout_flat = grad.transpose(2, 3, 0, 1).ravel()
        dX_col = np.zeros(self.output)
        dX_col[self.max_idxs, np.arange(self.max_idxs.size)] = dout_flat
        dX = col2im_indices(dX_col, (batch_size * channel_size, 1, f1, f2), self.pool_size, self.pool_size, padding=0, stride=self.strides).astype('float32')

        # Reshape back to match the input dimension: 5x10x28x28
        dX = dX.reshape(self.input_size)
        # if self.channel_first is False:
        #     x_reshaped = np.rollaxis(dX, 3, 1)
        #     return x_reshaped
        return dX

class Dense(Layers):

    def __init__(self, input_dim, output_dim, l2_regularization=None):
        # self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim/2.0)

        # glorot uniform
        self.W = np.random.uniform(low=-np.sqrt(6/(input_dim + output_dim)),
            high=np.sqrt(6/(input_dim + output_dim)),
            size=(input_dim, output_dim))
        # print(self.W.shape)
        self.b = np.zeros(output_dim).astype('float32')
        self.l2_regularization = l2_regularization

    def set_optimizer(self, optimizer):
        self.w_optimizer = optimizer.copy()
        self.b_optimizer = optimizer.copy()

    def forward(self, x):
        self.inputs = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad):      
        # print(grad)  
        self.d_w = np.mean(np.matmul(self.inputs[:, :, None], grad[:, None, :]), axis=0)
        if self.l2_regularization is not None:
            self.d_w += self.l2_regularization * ( self.W)
        self.d_b = np.mean(grad)
        # del self.inputs
        return np.dot(grad, self.W.T)

    def update(self):
        # print(self.d_w)
        self.W -= self.w_optimizer.update(self.d_w)
        self.b -= self.b_optimizer.update(self.d_b)

    def update_lr(self, lr):
        self.w_optimizer.update_lr(lr)
        self.b_optimizer.update_lr(lr)
    
    def has_weight(self):
        return True

class BatchNorm(Layers):
    '''Note this is only for inference design, 
    it do not implement backpropagation design
    '''
    def __init__(self, channel_size, running_mean=None, running_var=None, weight=None, bias=None, eps=1e-3, momentum=0.1):
        self.running_mean = running_mean if running_mean else np.zeros(channel_size)
        self.running_var = running_var if running_var else np.ones(channel_size)
        self.weight = weight if weight else np.zeros(channel_size)
        self.bias = bias if bias  else np.zeros(channel_size)
        self.eps = eps
        self.channel_size = channel_size
        self.momentum = momentum

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError('Invaild input size')
        x_channel_last = np.transpose(x, (0,2,3,1))
        shifted = (x_channel_last-self.running_mean) / np.sqrt(self.running_var)
        output = shifted*self.weight + self.bias

        return np.transpose(output, (0, 3, 1, 2)) # convert back to original code


class Dropout(Layers):

    def __init__(self, rates=0.1):
        self.dropout_rate = rates

    def forward(self, x, training=True):
        self.training = training
        if training:
            self.drop_mask = np.random.binomial(1, self.dropout_rate, size=x.shape) / self.dropout_rate
            output = x * self.drop_mask
            return output
        return x * (1-self.dropout_rate)

    def backward(self, grad):
        if self.training:
            return grad * self.drop_mask
        else:
            return grad

class Sigmoid(Layers):

    def forward(self, x):
        self.output = np.exp(x)/(1.0+np.exp(x))
        return self.output

    def __call__(self, inputs):
        self.inputs = inputs

    def backward(self, grad):
        output = grad * ( self.output - self.output**2)
        return output
    
class Softmax(Layers):

    def forward(self, x):
        # exps = np.exp(x - np.max(x)) # stable softmax?
        # self.output = exps / np.expand_dims(exps.sum(axis=1),axis=1)
        # return self.output
        kw = dict(axis=-1, keepdims=True)

        # make every value 0 or below, as exp(0) won't overflow
        xrel = x - x.max(**kw)

        exp_xrel = np.exp(xrel)
        self.output = exp_xrel / exp_xrel.sum(**kw)  
        return self.output

    def backward(self, grad):
        sum_softmax = np.sum(grad*self.output, axis=1)[:, None]
        return self.output*grad - self.output*sum_softmax

class ReLU(Layers):

    def forward(self, x):
        self.input = x
        return np.clip(x, 0, None)
    
    def backward(self, grad):
        return np.where(self.input > 0, grad,  0)

class Tanh(Layers):
    def forward(self, x):
        self.output = 2.0/(1.0+np.exp(-2*x)) - 1.0
        return self.output

    def __call__(self, inputs):
        self.inputs = inputs
