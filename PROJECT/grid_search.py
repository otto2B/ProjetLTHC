import torch
import tensorflow as tf
import numpy as np
import util as f
import gradient_descent as gd
import file_util as file_
#Size of the matrix Y
N = 2
M = N

#
list_lambda_ = [0.5, 0.75, 1, 1.25, 1.5]

#temperatures
list_beta_u = [1, 10, 30]
list_beta_v = []

#learning rates
list_lambda_1 = [0.1, 1, 10, 100]
list_lambda_2 = []

# Pas de temps
list_dt = [1, 1/2]

u_ = f.generate_vector(N)
v_ = f.generate_vector(M)

u_p_0 = f.generate_vector(N)
v_p_0 = f.generate_vector(M)

list_overlap = []
storing_params = []

for lambda_ in list_lambda_:

    Y = f.generate_Y(u_,v_, lambda_)

    for beta_u in list_beta_u:
        for lambda_1 in list_lambda_1:
            for dt in list_dt:

                M = N
                beta_v = beta_u
                lambda_2 = lambda_1

                u_p = u_p_0
                v_p = v_p_0

                (res_u, res_v) = gd.main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)
                
                list_overlap.append((res_u, res_v))
                storing_params.append((N, lambda_, beta_u, lambda_1, dt))

                file_.writeIntoFile(res_u.item(), res_v.item(), N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt)


# Calcule la précision (à combien l'overlap est loin de 1):

list_mean = []

for (overlap_u, overlap_v) in list_overlap:
    a = abs(1 - overlap_u)
    b = abs(1 - overlap_v)

    mean = (a+b)/2

    list_mean.append(mean)

# Trouve la plus petite moyenne (la paire d'overlap la plus proche de 1)

best_m = 1000
i = 0
for current_m in list_mean:
    if current_m < best_m:
        best_m = current_m
        index = i 
    i = i + 1

print("# RESULTAT :\n")
print("La meilleur moyenne des distances est: ", best_m)
print("Les overlaps sont : ", list_overlap[index][0], list_overlap[index][1])
print("Les paramètres étaient fixés à : ", "N=", storing_params[index][0],
        ", lambda_=", storing_params[index][1], ", beta_u=", storing_params[index][2],
        ", lambda_1=", storing_params[index][3], ", dt=", storing_params[index][4])