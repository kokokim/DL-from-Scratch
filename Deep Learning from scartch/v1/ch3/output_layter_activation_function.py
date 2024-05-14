#출력층 활성화 함수 구현(항등함수, 소프트맥스)
import numpy as np

def identity_function(x):
    return x

def softmax(a):
    c=np.max(a)
    exp_a=np.exp(a-c)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a

a=np.array([0.3, 2.9, 4.0])
y=softmax(a)
print(y)


