from typing import List, Union, Dict

from .np import np
# class MinMaxScalerError(Exception):
    
class Feature():
    def __init__(self):
        self.layers = [
            OneHotEncoding(),
            MinMaxScaler()
        ]

    def run(self, x, y):
        for layer in self.layers:
            x, y = layer.run(x, y)
        return x, y

class MinMaxScaler():
    def __init__(self, x_y:str = 'x') -> list:
        # try: 
        #     assert isinstance(index, list)
        # except AssertionError, TypeError as e:
        #     raise AssertionError('tpye is not {o}'.format(o='list'))
    
        self.min_max = None
        self.index = None
        self.x_y = x_y
    
    def run(self, x, y, index = None):
        if self.min_max is None:
            if self.x_y == 'x':
                self.min_max = np.zeros((x.shape[1],2))
                self.index = np.arange(0, x.shape[1], 1)
            else:
                self.min_max = np.zeros((y.shape[1],2))
                self.index = np.arange(0, y.shape[1], 1)


            if not index is None:
                self.index = index
            for i in self.index:
                self.min_max[i][0] = x[:, i].min()
                self.min_max[i][1] = x[:, i].max() - x[:, i].min()
        
        for i in self.index:
            x[:, i] = (x[:, i] - self.min_max[i][0]) / self.min_max[i][1]
        return x, y

class OneHotEncoding():
    def __init__(self):
        self.unique = None
    
    def run(self, x, y):
        if y is None:
            return x, y
        
        if self.unique is None:
            self.unique = np.unique(y)
        y = np.array([[1 if row == onehot else 0 for onehot in self.unique] for row in y])
        return x, y

