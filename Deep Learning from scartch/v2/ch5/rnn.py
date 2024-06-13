#rnn 클래스의 초기화와 순전파 메서드 구현
import numpy as np

class RNN:
    def __init__(self, Wx, Wh, b):
        self.params=[Wx, Wh, b]
        self.grads=[np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.chche=None

    def forward(self, x, h_prev):
        Wx, Wh, b=self.params
        t=np.matmul(h_prev, Wh)+np.matmul(x, Wx)+b
        h_next=np.tanh(t)

        self.cache=(x, h_prev, h_next)
        return h_next