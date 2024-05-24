# 4번째, 5번째 줄 있어야 v2/ch2에서 실행됨

import sys, os
sys.path.append(os.pardir)
from dataset.ptb import load_data
from preprocessing_with_python import preprocess
from statistical_based import create_co_matrix
from ppmi import ppmi
from ranking import most_similar
import numpy as np

window_size=2
wordvec_size=100

corpus, word_to_id, id_to_word=load_data('train')
# print('말뭉치 크기', len(corpus))
# print('corpus[:30]', corpus[:30])
# print()
# print('id_to_word[0]', id_to_word[0])
# print('id_to_word[1]', id_to_word[1])
# print('id_to_word[2]', id_to_word[2])
# print()
# print("word_to_id['car']", word_to_id['car'])
# print("word_to_id['happy']", word_to_id['happy'])
# print("word_to_id['lexus]", word_to_id['lexus'])

vocab_size=len(word_to_id)
print('동시발생 수 계산 ...')
C=create_co_matrix(corpus, vocab_size, window_size)
print('ppmi 계산')
W=ppmi(C, verbose=True)

print('svd 계산')
try:
    #truncated SVD(빠르당)
    from sklearn.utils.extmath import randomized_svd
    U,S,V=randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

except ImportError:
    #SVD(느리다)
    U,S,V=np.linalg.svd(W)

word_vecs=U[:, :wordvec_size]

querys=['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)




