import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import numpy as np

import keras.backend as K
from keras.layers import TimeDistributed, Embedding, Lambda, Input, Reshape, Activation, Dense
from keras.layers import merge
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

inputs_train  = np.load('data/babi/inputs_train.npy')
queries_train = np.load('data/babi/queries_train.npy')
answers_train = np.load('data/babi/answers_train.npy')
inputs_test   = np.load('data/babi/inputs_test.npy')
queries_test  = np.load('data/babi/queries_test.npy')
answers_test  = np.load('data/babi/answers_test.npy')

num_stories, story_maxsents, story_maxlen = inputs_train.shape
_, query_maxlen = queries_train.shape
vocab_size = max([inputs_train.max(), answers_train.max()]) + 1

inps = [inputs_train, queries_train]
val_inps = [inputs_test, queries_test]

# ulabs  = set(answers_train.squeeze())
# lookup = dict(zip(ulabs, range(len(ulabs))))
# answers_train = np.array([lookup[a] for a in answers_train.squeeze()]).reshape(-1, 1)
# answers_test  = np.array([lookup[a] for a in answers_test.squeeze()]).reshape(-1, 1)
# num_classes = len(ulabs)

# --
# Model

# emb_dim = 32
# hidden_sents = 10
# hidden_sents = story_maxsents

# def emb_sent_bow(inp):
#     emb = TimeDistributed(Embedding(vocab_size, emb_dim))(inp)
#     emb = Lambda(lambda x: K.sum(x, 2))(emb)
    
#     # Mixed information between sentence locations
#     emb = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(emb)
#     emb = TimeDistributed(Dense(hidden_sents))(emb)
#     emb = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)))(emb)
    
#     return emb


# inp_story = Input((story_maxsents, story_maxlen))
# emb_story = emb_sent_bow(inp_story)

# inp_q = Input((query_maxlen,))
# emb_q = Embedding(vocab_size, emb_dim)(inp_q)
# emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)
# emb_q = Reshape((1, emb_dim))(emb_q)

# x = merge.dot([emb_story, emb_q], axes=2)
# x = Reshape((hidden_sents,))(x)
# x = Activation('softmax')(x)
# match = Reshape((hidden_sents,1))(x)

# emb_c = emb_sent_bow(inp_story)
# x = merge.dot([match, emb_c], axes=1)
# response = Reshape((1, emb_dim))(x)
# response = merge.add([response, emb_q])
# response = Reshape((emb_dim, ))(x)
# response = Dense(vocab_size, activation='softmax')(response)

# answer = Model([inp_story, inp_q], response)

# answer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# K.set_value(answer.optimizer.lr, 5e-3)
# hist = answer.fit(inps, answers_train, verbose=2, epochs=20, batch_size=32, validation_data=(val_inps, answers_test))

# --
# Two hop model

emb_dim = 20

def emb_sent_bow(inp):
    emb_op = TimeDistributed(Embedding(vocab_size, emb_dim))
    emb    = emb_op(inp)
    emb    = Lambda(lambda x: K.sum(x, 2))(emb)
    return emb, emb_op

h = Dense(emb_dim)

def one_hop(u, A):
    C, _ = emb_sent_bow(inp_story)
    x = Reshape((1, emb_dim))(u)
    x = merge.dot([A, x], axes=2)
    x = Reshape((story_maxsents,))(x)
    x = Activation('softmax')(x)
    match = Reshape((story_maxsents,1))(x)
    
    x = merge.dot([match, C], axes=1)
    x = Reshape((emb_dim,))(x)
    x = h(x)
    x = merge.add([x, emb_q])
    return x, C

inp_story = Input((story_maxsents, story_maxlen))
inp_q     = Input((query_maxlen,))

emb_story, emb_story_op = emb_sent_bow(inp_story)

emb_q = emb_story_op.layer(inp_q)
emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)

response, emb_story = one_hop(emb_q, emb_story)
response, emb_story = one_hop(response, emb_story)
response, emb_story = one_hop(response, emb_story)
response = Dense(vocab_size, activation='softmax')(response)

answer = Model([inp_story, inp_q], response)
answer.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

K.set_value(answer.optimizer.lr, 5e-3)
hist = answer.fit(inps, answers_train, epochs=8, batch_size=32, validation_data=(val_inps, answers_test))