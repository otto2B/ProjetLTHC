import torch
import tensorflow as tf
import numpy as np
import functions as f

import csv
import sys

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

    return(u_p, v_p, Y, u_, v_)

#gradient version 1

def main_gradient_1_sans_proj(u_p, v_p, Y, u_, v_):

    for i in range(N):
        u_n = u_p - 1/lambda_1 * f.gradient_u_1(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.normal(0, dt, size=(1, N)).float() - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * f.gradient_v_1(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.normal(0, dt, size=(1, M)).float() - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u: ", res_u)
    print("g1_v: ", res_v)

    return (res_u, res_v)

def main_gradient_1_avec_proj(u_p, v_p, Y, u_, v_):

    for i in range(N):
        u_n = u_p - 1/lambda_1 * torch.transpose(torch.mm(f.proj(u_p,N),torch.transpose(f.gradient_u_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.transpose(torch.mm(f.proj(u_p,N),torch.normal(0, dt, size=(N, 1)).float()),0,1) - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * torch.transpose(torch.mm(f.proj(v_p,M),torch.transpose(f.gradient_v_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.transpose(torch.mm(f.proj(v_p,M),torch.normal(0, dt, size=(M, 1)).float()),0,1) - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_proj_u: ", res_u)
    print("g1_proj_v: ", res_v)

    return (res_u, res_v)

# gradient version 2

def main_gradient_2_sans_proj(u_p, v_p, Y, u_, v_):

    for i in range(N):
        u_n = u_p - 1/lambda_1 * f.gradient_u_2(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.normal(0, dt, size=(1, N)).float() - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * f.gradient_v_2(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.normal(0, dt, size=(1, M)).float() - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u: ", res_u)
    print("g2_v: ", res_v)

    return (res_u, res_v)

def main_gradient_2_avec_proj(u_p, v_p, Y, u_, v_):

    for i in range(N):
        u_n = u_p - 1/lambda_1 * torch.transpose(torch.mm(f.proj(u_p,N),torch.transpose(f.gradient_u_2(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.transpose(torch.mm(f.proj(u_p,N),torch.normal(0, dt, size=(N, 1)).float()),0,1) - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * torch.transpose(torch.mm(f.proj(v_p,M),torch.transpose(f.gradient_v_2(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.transpose(torch.mm(f.proj(v_p,M),torch.normal(0, dt, size=(M, 1)).float()),0,1) - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_proj_u: ", res_u)
    print("g2_proj_v: ", res_v)

    return (res_u, res_v)

# ecrit les résultats dans un fichier csv
# Il faut spécifier le chemin du fichier dans path.
def writeIntoFile(res_g1, res_g2):
    path = r"C:\Users\Admin\Desktop\Bachelor Project\data.xls"

    f = open(path, 'a', newline = "")

    w1 = []

    w1.append(N)
    w1.append(M)
    w1.append(lambda_)
    w1.append(beta_u)
    w1.append(beta_v)
    w1.append(lambda_1)
    w1.append(lambda_2)
    w1.append(dt)
    w1.append(res_g1[0].item())
    w1.append(res_g1[1].item())
    w1.append(res_g2[0].item())
    w1.append(res_g2[1].item())

    with f:
        writer = csv.writer(f)
        writer.writerow(w1)

# ===========================
# exemple: python test.py true 30

if len(sys.argv) != 3:
    print("python test.py $1 $2\n")
    print("$1 = true (avec les projecteurs) ou false (sans les projecteurs)\n")
    print("$2 = int, combien de fois on run les tests\n")
else:
    how_many_times = int(sys.argv[2])
    use_projector = sys.argv[1] == 'true'
    for i in range(how_many_times):
        (u, v, Y, u_, v_) = main()
        (u_2, v_2, Y_2) = (u, v, Y)

        if use_projector:
            writeIntoFile(main_gradient_1_avec_proj(u, v, Y, u_, v_), main_gradient_2_avec_proj(u_2, v_2, Y_2, u_, v_))
        else:
            writeIntoFile(main_gradient_1_sans_proj(u, v, Y, u_, v_), main_gradient_2_sans_proj(u_2, v_2, Y_2, u_, v_))