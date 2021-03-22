import torch
import tensorflow as tf
import numpy as np

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


# genere un vecteur sur l'hypersphere
def generateVector():
    x = torch.normal(0, 1, size=(1, N))
    x = x / torch.linalg.norm(x)
    x = x * torch.sqrt(tensor_N)
    return x

# calcul l'overlap entre deux tenseurs.
def overlap(x_etoile, x):
    return torch.dot(x_etoile, x)/N


def generate_Y(u_*,v_*): 
	uv = torch.dot(torch.transpose(u_*,0,1),v_*)
	eta = torch.normal(0, 1, size=(N, N))
	return uv + np.sqrt(N/lambda_)*eta
	
def gradient_v_1(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(N):
			sum += u_[i]*(Y[i][d].item()-u_[i]*v_[d])
		x.append(-2/(n**2)*sum)
	return torch.tensor(x)
	
def gradient_u_1(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(N):
			sum += v_[i]*(Y[d][i].item()-u_[d]*v_[i])
		x.append(-2/(n**2)*sum)
	return torch.tensor(x)	
	

def gradient_v_2(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(N):
			sum += u_[i]*(Y[i][d].item()-np.sqrt(lambda_/N)*u_[i]*v_[d])
		x.append(-np.sqrt(lambda_/N))*sum)
	return torch.tensor(x)
	
def gradient_u_2(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(N):
			sum += v_[i]*(Y[d][i].item()-np.sqrt(lambda_/N)u_[d]*v_[i])
		x.append(-np.sqrt(lambda_/N))*sum)
	return torch.tensor(x)	
	
	
	
	
	
	
		