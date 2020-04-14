from LoadData import load_data

from jmodel.util import *
from jmodel.Optimizers.BiascOptimizer import *
from jmodel.Models.Rnnlm import *
from jmodel.Trainers.RnnTrainer import *
# 設定超參數
batch_size = 20
wordvec_size = 100
hidden_size = 100  # RNN隱藏狀態向量的元素數
time_size = 35  # 展開RNN的大小
lr = 20.0
max_epoch = 50
max_grad = 0.25

# 載入學習資料
corpus, word_to_id, id_to_word = load_data("lotr.txt")
corpus_test = corpus[len(corpus)/5*4:]
corpus = corpus[:len(corpus)/5*4]
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# 產生模型
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# 套用梯度裁減並學習
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad,
            eval_interval=20)
trainer.plot(ylim=(0, 500))

# 用測試資料評估
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# 儲存參數
model.save_params()