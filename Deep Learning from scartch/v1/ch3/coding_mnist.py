#손글씨 숫자 인식
#train set을 이용해서 학습하지 않고 그냥 test set을 사용하여 바로 predict함
#train을 하지 않아도 나오는구나...? 의잉..?
#cd v1/ch3로 들어가서 실행해야 실행됨

# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network=pickle.load(f)
        return network
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    c=np.max(x)
    exp_a=np.exp(x-c)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a
    
def predict(network, x):
    W1, W2, W3=network['W1'], network['W2'], network['W3']
    b1, b2, b3=network['b1'], network['b2'], network['b3']

    a1=np.dot(x, W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1, W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2, W3)+b3
    y=softmax(a3)

    return y

x, t=get_data()
network=init_network()
accuracy_cnt=0

# for i in range(len(x)):
#     y=predict(network, x[i])
#     p=np.argmax(y)
#     if p==t[i]:
#         accuracy_cnt+=1

#배치 사이즈 적용
batch_size=100

for i in range(0, len(x), batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network, x_batch)
    p=np.argmax(y_batch, axis=1)
    accuracy_cnt+=np.sum(p==t[i:i+batch_size])

print("accuracy "+str(float(accuracy_cnt)/len(x)))
