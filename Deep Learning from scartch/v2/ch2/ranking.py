# 검색어가 주어지면 그 검색어와 비슷한 단어를 유사도 순으로 출력
import numpy as np
from preprocessing_with_python import preprocess
from statistical_based import cos_similarity, create_co_matrix

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    #검색어를 꺼냄
    if query not in word_to_id:
        print('%s 찾을수없음.' % query)
        return
    
    print('query', query)
    
    query_id=word_to_id[query]
    query_vec=word_matrix[query_id]

    vocab_size=len(id_to_word)
    
    similarity=np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i]=cos_similarity(word_matrix[i], query_vec)
    
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
        
            
x=np.array([100, -20, 2])
# print(x.argsort()) #argsort() 메서드는 넘파이 배열의 원소를 오름차순으로 정리함

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)
