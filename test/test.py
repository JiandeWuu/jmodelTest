import pickle

from LoadData import load_data
from jmodel.np import *
from jmodel.util import *
from jmodel.Function import *
from jmodel.Models.Rnnlm import *
from jmodel.Models.SimpleRnnlm import *
corpus, word_to_id, id_to_word = load_data("ptb.train.txt")
corpus = corpus[:50000]
vocab_size = len(word_to_id)
# test_word = ['it', 'has', 'no', 'bearing', 'on', 'our', 'work', 'force']
# tid = np.arange(len(test_word))
# for i in range(len(test_word)):
#     tid[i] = word_to_id[test_word[i]]

# tid = corpus[10265:10280]
# print(tid)
# print(id_to_word[int(corpus[10281])])
# test_data = to_gpu([tid])


wordvec_size = 100
hidden_size = 100  # RNN隱藏狀態向量的元素數

print("RNN")
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
model.load_params()
# 用測試資料評估
model.reset_state()
ppl_test = eval_perplexity(model, corpus)
print('test perplexity: ', ppl_test)

print("LSTM")
model = Rnnlm(vocab_size, 128, 128)
model.load_params("params/Rnnlm_b.pkl")
# 用測試資料評估
model.reset_state()
ppl_test = eval_perplexity(model, corpus)
print('test perplexity: ', ppl_test)

# model.reset_state()
# p = model.predict(test_data)
# s = softmax(p[0][len(test_word)-1])
# max_id_array = np.argpartition(s, -10)[-10:]
# for mid in max_id_array:
#     print(id_to_word[int(mid)] + "---" + str(s[int(mid)]))
