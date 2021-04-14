import torch
import tensorflow as tf
import numpy as np


# Genere un vecteur sur l'hypersphere
def generate_vector(d):
    x = torch.empty(d).normal_(mean=0,std=1)
    x = x / torch.linalg.norm(x)
    x = x * torch.sqrt(d)
    return x

# Calcul l'overlap entre deux tenseurs
def overlap(x_etoile, x, d):
    return x_etoile.dot(x)/d

# Genere la matrice Y
def generate_Y(N, M, u_,v_, lambda_): 
	uv = torch.tensordot(u_, v_, 0)
	eta = torch.normal(0, 1, size=(N.item(), M.item()))
	return torch.sqrt(lambda_/N)*uv + eta
	
# Return le projecteur associé au vecteur
def proj(vector,d):
	return torch.eye(d) - torch.tensordot(vector, vector, 0)/d

# GRADIENT N°1

def gradient_v_1(N, M, u_,v_,Y,lambda_):
	x = torch.tensordot(v_, u_, 0)
	x = torch.transpose(Y,0,1)-x
	x = torch.einsum('ij,j->i', x, u_)
	return (-2/(N.pow(2)))*x
	
def gradient_u_1(N, M, u_,v_,Y,lambda_):
	x = torch.tensordot(u_, v_, 0)
	x = Y-x
	x = torch.einsum('ij,j->i', x, v_)
	return (-2/(N.pow(2)))*x

# GRADIENT N°2

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