import torch
import tensorflow as tf
import numpy as np
import util as f
import gradient_descent as gd
import matplotlib.pyplot as plt
import file_util as fu

# Check if you can run the algo on your GPU
print("Is cuda available?")
if torch.cuda.is_available():  
  dev = "cuda:0" 
  print("Yes: ", torch.cuda.get_device_name(0))
else:  
  dev = "cpu"
  print("no: ", dev)
print()

# USE OR NOT GPU:
# comment or not the following line:
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# And set the right bool according to the previous line:
run_on_gpu = False

print("run on gpu: ", run_on_gpu)
print()

#Size of the matrix Y
N = torch.tensor(100)
M = torch.tensor(100)

# lambda
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
    #print(Y)
    
    # Conditions initiales
    u_p = f.generate_vector(N)
    v_p = f.generate_vector(M)
    
    # Perform the gradient descent
    # you can use with i=1 or 2:
    # main_gradient_i_avec_proj
    # main_gradient_i_sans_proj_normalisation
    # main_gradient_i_sans_proj
    (u_overlap, v_overlap) = gd.main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)

    # Do you want to write the result in a file ?
    #fu.writeIntoFile(abs(u_overlap).item(), abs(v_overlap).item(), N.item(), M.item(), lambda_.item(), beta_u.item(), beta_v.item(), lambda_1.item(), lambda_2.item(), dt.item())

    # Add the result to the global array
    global res_u
    global res_v
    res_u.append(abs(u_overlap))
    res_v.append(abs(v_overlap))


# LIST OF VALUE THAT WILL BE TESTED
#list_value_ = torch.tensor([0.0001, 0.0005, 0.001, 0.005,  0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.925, 0.95, 0.975, 1, 1.1, 1.15, 1.5, 1.2, 1.5, 1.7, 2, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200])
list_value_ = torch.arange(0.1, 1000, 0.1)

# Run the algo for all the given value
for value_to_test in list_value_:
    lambda_ = value_to_test

    print("> N=", N.item(), " M=", M.item(), " lambda=", lambda_.item(), " beta_u=", beta_u.item(), 
    " beta_v=", beta_v.item(), " lambda_1=", lambda_1.item(), " lambda_2=", lambda_2.item(), " dt=", dt.item(),)

    main()

# If we run on the gpu we need to copy back the tensor in the cpu
if(run_on_gpu):
    for i in range(len(res_u)):
        res_u[i] = res_u[i].to("cpu")
        res_v[i] = res_v[i].to("cpu")


# Plot the result
# linestyle=None or solid
# marker="" or "."
plt.plot(list_value_, res_v, linestyle = 'solid', marker= "", label="v_overlap")
plt.plot(list_value_, res_u, linestyle = 'solid', marker= "", label="u_overlap")
plt.legend()

plt.xlabel('lambda_')
plt.ylabel('overlap')

extraticks = [1]
plt.xticks(list(plt.xticks()[0]) + extraticks)
plt.show()