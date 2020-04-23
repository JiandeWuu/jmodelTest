import pickle

from LoadData import load_data
from jmodel.np import *
from jmodel.util import *
from jmodel.Function import *
from jmodel.Models.Rnnlm import *
corpus, word_to_id, id_to_word = load_data("ptb.train.txt")
tid = word_to_id['work']
test_data = to_gpu([[tid]])

# print(test_data.shape)
model = Rnnlm()
model.load_params
p = model.predict(test_data)
print(p[0][0])
s = softmax(p[0][0])
max_id_array = np.argpartition(s, -10)[-10:]
print(max_id_array)
for mid in max_id_array:
    print(id_to_word[int(mid)])