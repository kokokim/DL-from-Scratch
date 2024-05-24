# 차원 감소 중 svd 구현
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_with_python import preprocess
from statistical_based import create_co_matrix
from ppmi import ppmi   

text='you say goodbye and i say hello.'
corpus, word_to_id, id_to_word=preprocess(text)
vocab_size=len(word_to_id)
C=create_co_matrix(corpus, vocab_size, window_size=1)
W=ppmi(C)

U, S, V=np.linalg.svd(W) #svd는 넘파이의 linalg 모듈이 제공하는 svd 메서드로 실행할 수 있음

print(C[0]) #동시발생행렬
print(W[0]) #ppmi 행렬
print(U[0]) #svd

#희소벡터인 W[0]가 밀집벡터 U[0]로 변함

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()