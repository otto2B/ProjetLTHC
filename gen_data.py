import torch
import tensorflow as tf
import numpy as np
import functions as f
#import gradient_descent as gd

#Size of the matrix Y
N = 10
M = N

#
lambda_ = 1
data_size = 3

vectors_u = []
vectors_v = []
matrices = []

for i in range(data_size):
    u_ = f.generate_vector(N)
    v_ = f.generate_vector(M)
    
    Y = f.generate_Y(u_,v_, lambda_)
    vectors_u.append(u_)
    vectors_v.append(v_)
    matrices.append(Y)
    
vec_u = torch.Tensor(data_size, N)
vec_v = torch.Tensor(data_size, M)
mat = torch.Tensor(data_size, N, M)
torch.cat(vectors_u, out=vec_u)
torch.cat(vectors_v, out=vec_v)
torch.cat(matrices, out=mat)

print(vec_u)
print(vec_v)
print(mat)