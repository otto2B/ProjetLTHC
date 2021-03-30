import torch
import tensorflow as tf
import numpy as np


# genere un vecteur sur l'hypersphere
def generate_vector(d):
    x = torch.normal(0, 1, size=(1, d))
    x = x / torch.linalg.norm(x)
    x = x * torch.sqrt(torch.tensor(d))
    return x

# calcul l'overlap entre deux tenseurs.
def overlap(x_etoile, x, d):
    return torch.mm(x_etoile, torch.transpose(x,0,1))/d


def generate_Y(u_,v_, lambda_): 
	n = list(u_.shape)[1]
	m = list(v_.shape)[1]
	uv = torch.mm(torch.transpose(u_,0,1),v_)
	eta = torch.normal(0, 1, size=(n, m))
	return torch.sqrt(torch.tensor(n/lambda_))*uv + eta
	
def proj(vector,d):
	return torch.eye(d) - torch.mm(torch.transpose(vector,0,1),vector)/d
	
def gradient_v_1(u_,v_,Y):
	n = list(u_.shape)[1]
	m = list(v_.shape)[1]
	x = [[]]
	for d in range(m):
		sum = 0
		for i in range(n):
			sum += u_[0][i]*(Y[i][d].item()-u_[0][i]*v_[0][d])
		x[0].append(-2/(n*m)*sum)
	return torch.tensor(x)
	
def gradient_u_1(u_,v_,Y):
	n = list(u_.shape)[1]
	m = list(v_.shape)[1]
	x = [[]]
	for d in range(n):
		sum = 0
		for i in range(m):
			sum += v_[0][i]*(Y[d][i].item()-u_[0][d]*v_[0][i])
		x[0].append(-2/(n*m)*sum)
	return torch.tensor(x)	

def gradient_v_2(u_,v_,Y,lambda_):
	n = list(u_.shape)[1]
	m = list(v_.shape)[1]
	x = [[]]
	for d in range(m):
		sum = 0
		for i in range(n):
			sum += u_[0][i]*(Y[i][d].item()-np.sqrt(lambda_/n)*u_[0][i]*v_[0][d])
		x[0].append(-np.sqrt(lambda_/m)*sum)
	return torch.tensor(x)
	
def gradient_u_2(u_,v_,Y,lambda_):
	n = list(u_.shape)[1]
	m = list(v_.shape)[1]
	x = [[]]
	for d in range(n):
		sum = 0
		for i in range(m):
			sum += v_[0][i]*(Y[d][i].item()-np.sqrt(lambda_/n)*u_[0][d]*v_[0][i])
		x[0].append(-np.sqrt(lambda_/n)*sum)
	return torch.tensor(x)	