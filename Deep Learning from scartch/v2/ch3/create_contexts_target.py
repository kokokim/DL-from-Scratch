#말뭉치로부터 맥락과 타깃 만들어보자

import sys
import os
sys.path.append(os.pardir)
import numpy as np
from ch2.preprocessing_with_python import preprocess
from one_hot import convert_one_hot

text='you say goodbye and i say hello.'
corpus, word_to_id, id_to_word=preprocess(text)

def create_contexts_target(corpus, window_size=1):
    target=corpus[window_size:-window_size]
    contexts=[]

    for idx in range(window_size, len(corpus)-window_size):
        cs=[]
        for t in range(-window_size, window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)


#맥락과 타깃을 원핫표현으로 변환

contexts, target=create_contexts_target(corpus)
vocab_size=len(word_to_id)
target=convert_one_hot(target, vocab_size)
contexts=convert_one_hot(contexts, vocab_size)

print(target)
print(contexts)