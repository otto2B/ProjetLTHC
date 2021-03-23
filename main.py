import torch
import tensorflow as tf
import numpy as np
import functions as f

#Size of the matrix Y
N = 10
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

# Pas de temps
dt = 1/N

def main():
    u_ = f.generate_vector(N)
    v_ = f.generate_vector(M)
    
    Y = f.generate_Y(u_,v_)
    
    print(Y)
    
    # Conditions initiales
	
    u_p = f.generate_vector(N)
    v_p = f.generate_vector(M)
    
    for i in range(N):
        u_n = u_p - 1/lambda_1 * torch.transpose(torch.mm(f.proj(u_p,N),torch.transpose(f.gradient_u_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.transpose(torch.mm(f.proj(u_p,N),torch.normal(0, dt, size=(N, 1)).float()),0,1) - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * torch.transpose(torch.mm(f.proj(v_p,M),torch.transpose(f.gradient_v_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.transpose(torch.mm(f.proj(v_p,M),torch.normal(0, dt, size=(M, 1)).float()),0,1) - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        u_p = u_n
        v_p = v_n
    
    print(f.overlap(u_,u_n,N))
    print(f.overlap(v_,v_n,M))
    
    
    
    
    
    
main()

	
	
	
	
	
		