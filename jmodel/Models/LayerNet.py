import pickle


from ..np import *
from ..Layers.BiascLayers import *
from ..Optimizers.BiascOptimizer import *
from ..Layers.LossLayers import *

class ThreeLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        w1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        w2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        w3 = np.random.randn(H, H)
        b3 = np.random.randn(H) 
        self.layers = [
            Affine(w1, b1),
            Sigmoid(),
            Affine(w3, b3),
            Sigmoid(),
            # Affine(w3, b3),
            # Sigmoid(),
            Affine(w2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.t = []
        self.min = 0
        
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def fit(self, x, t, max_epoch = 10, batch_size = 32, optimizer = SGD(lr = 1)):
        data_size = len(x)
        max_iters = data_size // batch_size
        total_loss = 0
        loss_count = 0
        loss_list = []

        for epoch in range(max_epoch):
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                
                loss = self.forward(batch_x, batch_t)
                self.backward()
                optimizer.update(self.params, self.grads)

                total_loss += loss
                loss_count += 1

                if (iters + 1) % max_iters == 0:
                    avg_loss = total_loss / loss_count
                    print('| epoch %d | iter %d / %d | loss %.2f' % (epoch + 1, iters +1, max_iters, avg_loss))
                    loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0
        
        return loss_list
    
    def save_params(self, file_name = 'params/ThreeLayerNet.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name = 'params/ThreeLayerNet.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
    