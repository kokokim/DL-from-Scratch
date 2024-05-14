#다차원배열 (2차원은 행렬이라함)
import numpy as np

a=np.array([1,2,3,4,5])
b=np.array([[1,2],[3,4],[5,6]])
c=np.array([[1,2],[3,4]])
print(a)
print(b)
print(np.ndim(a))
print(np.ndim(b))
print(a.shape)
print(b.shape)

# 행렬의 곱(내적)
print(np.dot(b,c))
