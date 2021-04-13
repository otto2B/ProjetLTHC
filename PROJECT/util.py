import torch
import tensorflow as tf
import numpy as np


# genere un vecteur sur l'hypersphere
def generate_vector(d):
    x = torch.empty(d).normal_(mean=0,std=1)
    x = x / torch.linalg.norm(x)
    x = x * torch.sqrt(d)
    return x

# calcul l'overlap entre deux tenseurs.
def overlap(x_etoile, x, d):
    return x_etoile.dot(x)/d


def generate_Y(N, M, u_,v_, lambda_): 
	uv = torch.tensordot(u_, v_, 0)
	eta = torch.normal(0, 1, size=(N.item(), M.item()))
	return torch.sqrt(lambda_/N)*uv + eta
	
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

def gradient_v_2(N, M, u_,v_,Y,lambda_):
	x = torch.tensordot(v_, u_, 0)
	x = torch.transpose(Y,0,1)-(torch.sqrt(lambda_/N)*x)
	x = torch.einsum('ij,j->i', x, u_)
	return (-torch.sqrt(lambda_/N))*x
	
def gradient_u_2(N, M, u_,v_,Y,lambda_):
	x = torch.tensordot(u_, v_, 0)
	x = Y-(torch.sqrt(lambda_/N)*x)
	x = torch.einsum('ij,j->i', x, v_)
	return (-torch.sqrt(lambda_/N))*x