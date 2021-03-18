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
	v = MultivariateNormal(torch.zeros(N), torch.eye(N))
	v = v/v.norm(2)
	norm = np.random.normal
	normal_deviates = norm(size=(dim, N))

	radius = np.sqrt((normal_deviates**2).sum(axis=0))
	y = normal_deviates/radius
	return y



def overlap(u_etoile, u):
    return (u_etoile.dot(u)/N)

