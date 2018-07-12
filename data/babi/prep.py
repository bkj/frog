import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import re
import argparse
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# --
# Helpers

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        if int(nid) == 1: story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            substory = [[str(i)+":"]+x for i,x in enumerate(story) if x]
            data.append((substory, q, a))
            story.append('')
        else:
            story.append(tokenize(line))
        
    return data

def get_stories(f):
    data = parse_stories(open(f).readlines())
    return [(story, q, answer) for story, q, answer in data]

def flatten_stories(stories):
    for s in stories:
        for sent in s[0]:
            for word in sent:
                yield word
        for word in s[1]:
            yield word
        yield s[2]

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X  = []
    Xq = []
    Y  = []
    for story, query, answer in data:
        x  = [[word_idx[w] for w in s] for s in story]
        xq = [word_idx[w] for w in query]
        y  = [word_idx[answer]]
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    
    return (
        [pad_sequences(x, maxlen=story_maxlen) for x in X],
        pad_sequences(Xq, maxlen=query_maxlen), 
        np.array(Y)
    )

def stack_inputs(inputs):
    for i,it in enumerate(inputs):
        inputs[i] = np.concatenate([it, np.zeros((story_maxsents-it.shape[0],story_maxlen), 'int')])
    
    return np.stack(inputs)


# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_path = 'data/babi/simple-en-10k/qa%d_train.txt' % args.task_id
    test_path  = 'data/babi/simple-en-10k/qa%d_test.txt' % args.task_id

    train_stories = get_stories(train_path)
    test_stories = get_stories(test_path)

    stories = train_stories + test_stories
    story_maxlen   = max((len(s) for x, _, _ in stories for s in x))
    story_maxsents = max((len(x) for x, _, _ in stories))
    query_maxlen   = max(len(x) for _, x, _ in stories)

    vocab = sorted(set(flatten_stories(stories)))
    vocab.insert(0, '<PAD>')
    vocab_size = len(vocab)

    word_idx = dict((c, i) for i, c in enumerate(vocab))

    inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
    inputs_test, queries_test, answers_test    = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

    inputs_train = stack_inputs(inputs_train)
    inputs_test = stack_inputs(inputs_test)

    np.save('data/babi/inputs_train', inputs_train)
    np.save('data/babi/queries_train', queries_train)
    np.save('data/babi/answers_train', answers_train)
    np.save('data/babi/inputs_test', inputs_test)
    np.save('data/babi/queries_test', queries_test)
    np.save('data/babi/answers_test', answers_test)
