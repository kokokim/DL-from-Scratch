# 활성화 함수 구현(sigmoid, step, relu)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    h=1/(1+np.exp(-x))
    return h


def step_function(x):
    #넘파이 배열도 지원하도록 작성
    y=x>0
    return y.astype(int) #넘파이 배열의 자료형 변환할때 astype() 메서드 이용
    # false, true, true 가 0, 1, 1로 바뀜

def relu(x):
    return np.maximum(0, x) #maximum은 두 입력 값중 큰 값을 선택해 반환하는 함수

x=np.arange(-5.0, 5.0, 0.1) #np.arrange는 -5에서 5전까지 0.1간격의 넘파이 배열을 생성함
y1=step_function(x)
plt.plot(x,y1)
plt.ylim(-0.1, 1.1)
plt.show()

y2=sigmoid(x)
plt.plot(x, y2)
plt.show()

y3=relu(x)
plt.plot(x, y3)
plt.show()

