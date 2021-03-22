import torch
import tensorflow as tf
import numpy as np

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
	eta = torch.normal(0, 1, size=(N, M))
	return uv + np.sqrt(N/lambda_)*eta
	
def gradient_v_1(u_,v_,Y):
	x = []
	for d in range(M):
		sum = 0
		for i in range(N):
			sum += u_[i]*(Y[i][d].item()-u_[i]*v_[d])
		x.append(-2/(N*M)*sum)
	return torch.tensor(x)
	
def gradient_u_1(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(M):
			sum += v_[i]*(Y[d][i].item()-u_[d]*v_[i])
		x.append(-2/(N*M)*sum)
	return torch.tensor(x)	
	

def gradient_v_2(u_,v_,Y):
	x = []
	for d in range(M):
		sum = 0
		for i in range(N):
			sum += u_[i]*(Y[i][d].item()-np.sqrt(lambda_/N)*u_[i]*v_[d])
		x.append(-np.sqrt(lambda_/M))*sum)
	return torch.tensor(x)
	
def gradient_u_2(u_,v_,Y):
	x = []
	for d in range(N):
		sum = 0
		for i in range(M):
			sum += v_[i]*(Y[d][i].item()-np.sqrt(lambda_/N)u_[d]*v_[i])
		x.append(-np.sqrt(lambda_/N))*sum)
	return torch.tensor(x)	
	