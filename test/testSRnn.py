# coding: utf-8
from LoadData import load_data

from jmodel.util import *
from jmodel.Optimizers.BiascOptimizer import *
from jmodel.Models.SimpleRnnlm import *
from jmodel.Trainers.RnnTrainer import *


# 設定超參數
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNN的隱藏狀態向量的元素數
time_size = 5  # 展開RNN的大小
lr = 0.1
max_epoch = 1

# 載入學習資料
# 載入學習資料
corpus, word_to_id, id_to_word = load_data("lotr.txt")
corpus_test = corpus[len(corpus)//5*4:]
corpus = corpus[:len(corpus)//5*4]
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 產生模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size)
trainer.plot(ylim=(0, 1000))

# 儲存參數
model.save_params()