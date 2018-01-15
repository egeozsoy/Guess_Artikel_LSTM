from create_article_features import createFeature
from create_article_features import convertWordToNumber
import tflearn
import tensorflow
import numpy as np
import os



tensorflow.reset_default_graph()
train_x, train_y, test_x, test_y = createFeature(random_size=0.3)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

# h_dim = 128
npoch = 150
h_dim = 64
batch_size = 32
dropout = 0.7
learning_rate = 0.001

net = tflearn.input_data(shape=[None, 30])
net = tflearn.embedding(net, input_dim=len(train_x), output_dim=h_dim)
net = tflearn.lstm(net, h_dim, dropout=dropout)
net = tflearn.fully_connected(net, 3, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                         loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=3)

model_name = "model_npoch_{}_hdim_{}.model".format(str(npoch) , str(h_dim))
print(model_name)

#model reloading not needed right now

if os.path.isfile('./' + model_name + '.meta'):
    print("Model exists")
    model.load(model_name , weights_only= True)
else:
    model.fit(train_x, train_y, validation_set=(test_x, test_y), n_epoch= npoch, batch_size=batch_size, show_metric=True)
    model.save(model_name)




while True:

    newWord = convertWordToNumber(input('Enter a word').lower())
    pred = model.predict([newWord])
    print(pred)
    pred = pred[0]
    if pred[0] > pred[1] and pred[0] > pred[2]:
        print('der')
    elif pred[1] > pred[0] and pred[1] > pred[2]:
        print('die')
    else:
        print('das')
