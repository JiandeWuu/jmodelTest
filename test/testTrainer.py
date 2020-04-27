from LoadData import load_data

from jmodel.util import *
from jmodel.Optimizers.BiascOptimizer import *
from jmodel.Models.Rnnlm import *
from jmodel.Trainers.RnnTrainer import *

# 設定超參數
batch_size_array = [16, 32, 64, 128]
wordvec_size_array = [32, 128, 256, 512]
hidden_size_array = [32, 128, 256, 512]  # RNN隱藏狀態向量的元素數
time_size_array = [8, 16, 32, 64]  # 展開RNN的大小
lr_array = [0.1, 1, 10, 20.0]
max_epoch = 100
max_grad = 0.25

# 載入學習資料
corpus, word_to_id, id_to_word = load_data("ptb.train.txt")
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

best_model = None
best_ppl = None
try:
    for batch_size in batch_size_array:
        for wordvec_size in wordvec_size_array:
            for time_size in time_size_array:
                for lr in lr_array:
                    # 產生模型
                    model = Rnnlm(vocab_size, wordvec_size, wordvec_size)
                    optimizer = SGD(lr)
                    trainer = RnnlmTrainer(model, optimizer)

                    # 套用梯度裁減並學習
                    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
                    
                    model.reset_state()
                    ppl = eval_perplexity(model, corpus)
                    print('perplexity: ', ppl)
                    if best_ppl is None or ppl < best_ppl:
                        best_ppl = ppl
                        best_model = model
                        model.save_params("params/Rnnlm_b.pkl")

except :
    with open('params/stop.pickle', 'wb') as f:
        pickle.dump({'batch_size' : batch_size, 'wordvec_size' : wordvec_size, 'time_size' : time_size, 'lr' : lr, 'best_ppl' : best_ppl,'model' : model.params}, f)
        
finally:
    # 用測試資料評估
    best_model.reset_state()
    corpus_test, _, _ = load_data("ptb.test.txt")

    ppl_test = eval_perplexity(best_model, corpus_test)
    print('test perplexity: ', ppl_test)

    # 儲存參數
    best_model.save_params("params/Rnnlm_b.pkl")
