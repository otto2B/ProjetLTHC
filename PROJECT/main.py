import torch
import tensorflow as tf
import numpy as np
import util as f
import gradient_descent as gd
import matplotlib.pyplot as plt

#Size of the matrix Y
N = 100
M = N

#
lambda_ = 1

#temperatures
beta_u = float("inf")
beta_v = beta_u

#learning rates
lambda_1 = 1
lambda_2 = lambda_1

# Pas de temps
dt = 1/100

res = []

def main():
    u_ = f.generate_vector(N)
    v_ = f.generate_vector(M)
    
    Y = f.generate_Y(u_,v_, lambda_)
    
    #print(Y)
    
    # Conditions initiales
	
    u_p = f.generate_vector(N)
    v_p = f.generate_vector(M)
    
    (u_overlap, v_overlap) = gd.main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)
    
    res.append(abs(u_overlap))

    #print(u_overlap)
    #print(v_overlap)
    

list_lambda_ = [0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 1, 1.1, 1.15, 1.5, 1.2, 1.5, 1.7, 2, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]

    
for lambda_to_test in list_lambda_:
    lambda_ = lambda_to_test
    main()

for i in range(len(res)):
    plt.plot(list_lambda_[i], res[i], 'b.')

plt.xlabel('lambda')
plt.ylabel('overlap(u*,u)')
extraticks = [1]
plt.xticks(list(plt.xticks()[0]) + extraticks)
plt.show()
	
	
	
	
		