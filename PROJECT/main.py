import torch
import tensorflow as tf
import numpy as np
import util as f
import gradient_descent as gd

#Size of the matrix Y
N = 2
M = N

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
    
    Y = f.generate_Y(u_,v_, lambda_)
    
    print(Y)
    
    # Conditions initiales
	
    u_p = f.generate_vector(N)
    v_p = f.generate_vector(M)
    
    (u_overlap, v_overlap) = gd.main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)
    
    print(u_overlap)
    print(v_overlap)
    
    

main()

	
	
	
	
	
		