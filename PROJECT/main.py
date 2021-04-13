import torch
import tensorflow as tf
import numpy as np
import util as f
import gradient_descent as gd
import matplotlib.pyplot as plt
import file_util as fu

print("Is cuda available?")
if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("Yes: ", torch.cuda.get_device_name(0))
else:  
  dev = "cpu"
  print("no: ", dev)
print()

# USE OR NOT GPU:
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
run_on_gpu = False

print("run on gpu: ", run_on_gpu)
print()

#Size of the matrix Y
N = torch.tensor(100)
M = torch.tensor(100)

#
lambda_ = torch.tensor(10)

#temperatures
beta_u = torch.tensor(float("inf"))
beta_v = torch.tensor(float("inf"))

#learning rates
lambda_1 = torch.tensor(1)
lambda_2 = torch.tensor(1)

# Pas de temps
dt = torch.tensor(1/100)

res_u = []
res_v = []

def main():
    u_ = f.generate_vector(N)
    v_ = f.generate_vector(M)
    #print(u_.is_cuda)
    #print(v_.is_cuda)
    
    Y = f.generate_Y(N, M, u_,v_, lambda_)
    #print(Y.is_cuda)
    #print("u, v, Y")
    
    #print(Y)
    
    # Conditions initiales
	
    u_p = f.generate_vector(N)
    v_p = f.generate_vector(M)
    #print(u_p.is_cuda)
    
    (u_overlap, v_overlap) = gd.main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)

    #fu.writeIntoFile(abs(u_overlap), abs(v_overlap), N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)

    global res_u
    global res_v
    res_u.append(abs(u_overlap))
    res_v.append(abs(v_overlap))

    #print(u_overlap)
    #print(v_overlap)
    

list_lambda_ = torch.tensor([0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 1, 1.1, 1.15, 1.5, 1.2, 1.5, 1.7, 2, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200])
#list_lambda_ = torch.arange(1, 1000, 10)

#list_beta_u = torch.tensor([0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 50, 100, 200, 500, 1000, 5000, 10000])
#list_beta_u = torch.tensor([0, 1])
    
for lambda_to_test in list_lambda_:
    lambda_ = lambda_to_test

    print("> N=", N.item(), " M=", M.item(), " lambda=", lambda_.item(), " beta_u=", beta_u.item(), 
    " beta_v=", beta_v.item(), " lambda_1=", lambda_1.item(), " lambda_2=", lambda_2.item(), " dt=", dt.item(),)

    main()

if(run_on_gpu):
    for i in range(len(res_u)):
        res_u[i] = res_u[i].to("cpu")
        res_v[i] = res_v[i].to("cpu")

for i in range(len(res_u)):
    plt.plot(list_lambda_[i], res_u[i], 'b.')
    plt.plot(list_lambda_[i], res_v[i], 'r.')

plt.xlabel('lambda_')
plt.ylabel('overlap')
extraticks = [1]
plt.xticks(list(plt.xticks()[0]) + extraticks)
plt.show()

	
	
		