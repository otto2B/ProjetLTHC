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

def main():
    u_ = f.generate_vector(N)
    v_ = f.generate_vector(M)
    
    Y = f.generate_Y(u_,v_)
    
    print(Y)
    
    
    
    
main()

	
	
	
	
	
		