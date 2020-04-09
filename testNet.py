from sklearn.model_selection import train_test_split

from LoadData import *

from jmodel.util import *
from jmodel.Optimizers.BiascOptimizer import *
from jmodel.Models.LayerNet import *
from jmodel.Trainers.NetTrainer import *
# 設定超參數
batch_size_array = [32, 128, 256, 512] 
lr_array = [0.1, 0.5, 1]
max_epoch_array = [320]
hidden_size_array = [16, 64, 128]
# batch_size_array = [8]
# max_epoch_array = [1]
# hidden_size_array = [16, 32, 64, 128]


# 載入學習資料
x, t, out_to_value = load('regionwide.csv')
X_train, X_test, Y_train, Y_test = train_test_split(x, t, test_size=0.2, shuffle=20200316)

# x = x[:len(x)/5*4]
best_loss = None
for batch_size in batch_size_array:
    for max_epoch in max_epoch_array:
        for hidden_size in hidden_size_array:
            model = ThreeLayerNet(input_size=X_train.shape[1], hidden_size=hidden_size, output_size=len(out_to_value)) #, params=params)
            nn = Trainer(model, out_to_value)
            loss = nn.fit(X_train, Y_train, max_epoch = max_epoch, batch_size = batch_size)
            if best_loss is None or best_loss > loss:
                best_loss = loss
                best_batch_size = batch_size
                best_max_epoch = max_epoch
                best_hidden_size = hidden_size

                nn.save()

print('| batch_size %d | max_epoch %d | hidden_size %d | loss %.5f' % (best_batch_size, best_max_epoch, best_hidden_size, best_loss))

# 儲存參數
model.save_params()
