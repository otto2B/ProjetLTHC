import torch
import tensorflow as tf
import numpy

#Size of the matrix Y
N = 600
M = N
dim = 1

#
lambda_ = 1

#temperatures
beta_u = 1
beta_v = beta_u

#learning rates
lambda_1 = 1
lambda_2 = lambda_1


# genere une matrice sur l'hypersphere
def generateVector():
    x = torch.normal(0, 1, size=(1, N))
    x = x / torch.linalg.norm(x)
    x = x * torch.sqrt(tensor_N)
    return x

# calcul l'overlap entre deux tensors.
def overlap(x_etoile, x):
    return torch.dot(x_etoile, x)/N