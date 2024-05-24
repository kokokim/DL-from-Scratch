# 자연어 전처리
import numpy as np

text='You say goodbye and I say hello.'

text=text.lower()
text=text.replace('.', ' .')
# print(text)

words=text.split()
# print(words)

word_to_id={}
id_to_word={}

for word in words:
    if word not in word_to_id:
        new_id=len(word_to_id)
        word_to_id[word]=new_id
        id_to_word[new_id]=word

# print(id_to_word)
# print(word_to_id)

#파이썬 내포 표기를 사용하여 단어 목록에서 단어id목록으로 변환한 다음, 다시 넘파이 배열로 변환
corpus=[word_to_id[w] for w in words]
corpus=np.array(corpus)
# print(corpus)

#합쳐보자
def preprocess(text):
    text=text.lower()
    text=text.replace('.', ' .')
    words=text.split()

    word_to_id={}
    id_to_word={}

    for word in words:
        if word not in word_to_id:
            new_id=len(word_to_id)
            word_to_id[word]=new_id
            id_to_word[new_id]=word

    corpus=np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

# print(preprocess(text))