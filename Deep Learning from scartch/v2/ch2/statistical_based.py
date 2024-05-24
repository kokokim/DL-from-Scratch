import sys
sys.path.append('..')
import numpy as np
from preprocessing_with_python import preprocess

text='you say goodbye and i say hello.'
corpus, word_to_id, id_to_word=preprocess(text)

#동시발생행렬 구현
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size=len(corpus)
    co_matrix=np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus): #corpus의 값과 그에 따른 인덱스값도 함께 출력됨
        for i in range(1, window_size+1):
            left_idx=idx-i
            right_idx=idx+i

            if left_idx>=0:
                left_word_id=corpus[left_idx]
                co_matrix[word_id, left_word_id]+=1

            if right_idx<corpus_size:
                right_word_id=corpus[right_idx]
                co_matrix[word_id, right_word_id]+=1

    return co_matrix


def cos_similarity(x,y, eps=1e-8):
    nx=x/(np.sqrt(np.sum(x**2))+eps)
    ny=y/(np.sqrt(np.sum(y**2))+eps)
    return np.dot(nx, ny)

# you와 i의 유사도 구하기
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size)

c0=C[word_to_id['you']]
c1=C[word_to_id['i']]
# print(cos_similarity(c0, c1))