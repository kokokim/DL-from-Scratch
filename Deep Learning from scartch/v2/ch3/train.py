import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size=1
hidden_size=5
batch_size=3
max_epoch=1000

text='you say goodbye and i say hello.'
corpus, word_to_id, id_to_word=preprocess(text)

vocab_size=len(word_to_id)
contexts, target=create_contexts_target(corpus, window_size)
target=convert_one_hot(target, vocab_size)
contexts=convert_one_hot(contexts, vocab_size)

model=SimpleCBOW(vocab_size, hidden_size)
optimizer=Adam()
trainer=Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

#마침내 단어를 밀집벡터로 나타냄. 이 벡터가 바로 단어의 분산 표현이다!
word_vecs=model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])