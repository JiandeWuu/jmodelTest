#shuffle
import random
import pickle
import datetime

from ..np import *
from ..util import *
from ..Function import * 
from ..Layers.LossLayers import *
from ..Layers.BiascLayers import *
from ..Feature.BiascFeature import *
from ..Optimizers.BiascOptimizer import *

class Trainer():
    def __init__(self, model, out_to_value = None):
        self.model = model
        self.feature = Feature()
        self.out_to_value = out_to_value

    def fit(self, x, t, max_epoch = 10, batch_size = 32, learning_rate = 1):
        x, t = self.feature.run(x, t)
        
        optimizer = SGD(lr = learning_rate)

        model = self.model
        loss = model.fit(x, t, max_epoch, batch_size, optimizer)
        
    def loss(self, x, t):
        x, t = self.feature.run(x, t)
        return self.model.forward(x, t)
    
    def predict(self, x):
        x = np.array(x)
        x, t = self.feature.run(x, None)
        score = self.model.predict(x)
        y = softmax(score)
        t = np.argmax(y, axis=1)
        if not self.out_to_value is None:
            return self.out_to_value[t]
        return t
    
    def save(self, file_name = 'NetTrainer.pickle'):
        with open('params/' + file_name, 'wb') as f:
            pickle.dump(self, f)
