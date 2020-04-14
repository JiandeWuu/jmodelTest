import pickle
with open('params/NetTrainer.pickle', 'rb') as f:
    trainer = pickle.load(f)
test_data = [[121.5934036,25.0771788],[121.5048083, 25.0470229],[0.5, 0.5]]
print(trainer.predict(test_data))
print(1)