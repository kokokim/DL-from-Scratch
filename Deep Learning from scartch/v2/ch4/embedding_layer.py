import numpy as np

class Embedding:
    def __init__(self, W):
        self.params=[W]
        self.grads=[np.zeros_like(W)]
        self.idx=None

    def forward(self, idx):
        W, =self.params
        self.idx=idx
        out=W[idx]
        return out
    
    def backward(self, dout):
        dW,=self.grads
        dW[...]=0
        # dW[self.idx]=dout #이렇게 하면 중복문제 발생
        for i, word_id in enumerate(self.idx):
            print(i, ", ", word_id)
            dW[word_id]+=dout[i] #중복발생시 더하는걸로

        #np.add.at(A, idx, B) #B를 A의 idx번째 행에 더해줌
        return None