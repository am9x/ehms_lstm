import tensorflow as tf
import numpy as np
print(tf.VERSION)
from tensorflow.keras.layers import LSTM,Dense,CuDNNLSTM

print(tf.keras.__version__)

lookback=15
lookforward=1#目前只支持预测一步
features=400
out_dim=400
optimizer = tf.train.AdamOptimizer()
model = tf.keras.Sequential()
#model.add(Dense(128,input_shape=(lookback, features)))

model.add(CuDNNLSTM(1024, return_sequences=True,input_shape=(lookback, features)))
model.add(CuDNNLSTM(2048))
model.add(Dense(out_dim))
model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy','mae'])

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='mse',
#               metrics=['accuracy','loss'])

# estimator = tf.keras.estimator.model_to_estimator(model)

strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
keras_estimator = tf.keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/home/admin0/william/ehms_lstm/tmp/lstm_model_dirfrom_generator1')

from gensim.models import Word2Vec
word2vec_model = Word2Vec.load("/home/admin0/gitproject/word2vec-Chinese/data/cate1/word2vec.model")

import glob
import pickle as pkl
text=''
for file in glob.glob('/home/admin0/gitproject/word2vec-Chinese/data/cate1/seg.txt'):
    print(file)
    with open(file,'r')as f:
        text = text+f.read()+"\n"
word_vec_length = word2vec_model.vector_size

words=text.split(" ")
len(words)

def gen_index(words_length):
    start = -1  
    while True:
        start=start+1
        if(start>=words_length):
            start=0 
        yield start

def getvecs_by_idx(start):
    idx = start
    wd_list=[]
    n_words = lookback+lookforward
    vecs = np.zeros((n_words,features),dtype=np.float32)
    j=0
    while len(wd_list)<n_words:
        if idx>=len(words):
            idx=0
        try:
            vd = words[idx]
            vec=(word2vec_model.wv[vd])
            wd_list.append(vd)
            vecs[j]=vec
            j=j+1
        except:
            vec=np.zeros((word_vec_length))
        idx=idx+1
        #print("vecs:",vecs.shape)
    dataXY=vecs
    return dataXY[:lookback,:],dataXY[lookback:,:].reshape((features*lookforward))
def set_shape(datax,datay):
    datax.set_shape([ lookback, features])
    datay.set_shape([ features])

    return datax,datay

batch_size=2048
def input_fn_2():
    dataset = tf.data.Dataset.from_generator(lambda:gen_index(10),(tf.int32))
    print("dataset.output_shapes",dataset.output_shapes)

    dataset = dataset.map(
        lambda start,: tuple(tf.py_func(
            getvecs_by_idx, [start], [tf.float32, tf.float32])),
        num_parallel_calls=4
    )
    dataset = dataset.map(set_shape)

    print("dataset.output_shapes",dataset.output_shapes)
#     dataset=dataset.prefetch(1024)
#     dataset = dataset.shuffle(2048).repeat().batch(batch_size)
    #dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    dataset=dataset.prefetch(2)

    return dataset


dataset = input_fn_2()

print("dataset.output_shapes",dataset.output_shapes)

sess=tf.Session()

iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
for i in range(15):
    idx=sess.run(one_element)
    print(idx[0].shape,idx[1].shape)


#keras_estimator.train(input_fn=input_fn_2, steps=2000000)
