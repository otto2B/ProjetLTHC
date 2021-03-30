import torch
import tensorflow as tf
import numpy as np
import util as f

iteration = 1000

def main_gradient_1_sans_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        u_n = u_p - 1/lambda_1 * f.gradient_u_1(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.normal(0, np.sqrt(dt), size=(1, N)).float() - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * f.gradient_v_1(u_p,v_p,Y) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.normal(0, np.sqrt(dt), size=(1, M)).float() - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u: ", res_u)
    print("g1_v: ", res_v)

    return (res_u, res_v)

def main_gradient_1_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        sqrt_dt = torch.tensor(torch.sqrt(dt))

        u_n = u_p - 1/lambda_1 * f.gradient_u_1(u_p,v_p,Y) * dt + torch.sqrt(2/(lambda_1*beta_u)) * torch.normal(0, sqrt_dt, size=(1, N)).float() - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * f.gradient_v_1(u_p,v_p,Y) * dt + torch.sqrt(2/(lambda_2*beta_v)) * torch.normal(0, sqrt_dt, size=(1, M)).float() - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        
        u_n = u_n / torch.linalg.norm(u_n)
        v_n = v_n / torch.linalg.norm(v_n)
        u_n = u_n * torch.sqrt(torch.tensor(N))
        v_n = v_n * torch.sqrt(torch.tensor(N))
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u_norm: ", res_u)
    print("g1_v_norm: ", res_v)

    return (res_u, res_v)

def main_gradient_1_avec_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        u_n = u_p - 1/lambda_1 * torch.transpose(torch.mm(f.proj(u_p,N),torch.transpose(f.gradient_u_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.transpose(torch.mm(f.proj(u_p,N),torch.normal(0, np.sqrt(dt), size=(N, 1)).float()),0,1) - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * torch.transpose(torch.mm(f.proj(v_p,M),torch.transpose(f.gradient_v_1(u_p,v_p,Y), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.transpose(torch.mm(f.proj(v_p,M),torch.normal(0, np.sqrt(dt), size=(M, 1)).float()),0,1) - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u_proj: ", res_u)
    print("g1_v_proj: ", res_v)

    return (res_u, res_v)

# gradient version 2

def main_gradient_2_sans_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        u_n = u_p - 1/lambda_1 * f.gradient_u_2(u_p,v_p,Y,lambda_) * dt + np.sqrt(2/(lambda_1*beta_u)) * torch.normal(0, np.sqrt(dt), size=(1, N)).float() - (N-1)/(N*lambda_1*beta_u)*u_p*dt
        v_n = v_p - 1/lambda_2 * f.gradient_v_2(u_p,v_p,Y,lambda_) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.normal(0, np.sqrt(dt), size=(1, M)).float() - (M-1)/(M*lambda_2*beta_v)*v_p*dt
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u: ", res_u)
    print("g2_v: ", res_v)

    return (res_u, res_v)

def main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        sqrt_dt = torch.tensor(np.sqrt(dt))

        u_1 = (1/lambda_1) * f.gradient_u_2(u_p,v_p,Y,lambda_) * dt
        u_2 = torch.sqrt(torch.tensor(2/(lambda_1*beta_u))) * torch.normal(0, sqrt_dt, size=(1, N))
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt

        #print(u_1, " ", u_2, " ", u_3)

        u_n = u_p - u_1 + u_2 - u_3

        v_1 = 1/lambda_2 * f.gradient_v_2(u_p,v_p,Y,lambda_) * dt
        v_2 = torch.sqrt(torch.tensor(2/(lambda_2*beta_v))) * torch.normal(0, sqrt_dt, size=(1, M))
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3
        u_n = u_n / torch.linalg.norm(u_n)
        v_n = v_n / torch.linalg.norm(v_n)
        u_n = u_n * torch.sqrt(torch.tensor(N))
        v_n = v_n * torch.sqrt(torch.tensor(N))
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u_norm: ", res_u, "g2_v_norm: ", res_v)

    return (res_u, res_v)

def main_gradient_2_avec_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        u_1  = (1/lambda_1) * torch.transpose(torch.mm(f.proj(u_p,N),torch.transpose(f.gradient_u_2(u_p,v_p,Y,lambda_), 0, 1)), 0, 1) * dt
        u_2 = np.sqrt(2/(lambda_1*beta_u)) * torch.transpose(torch.mm(f.proj(u_p,N),torch.normal(0, np.sqrt(dt), size=(N, 1))),0,1)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt

        u_n = u_p - u_1 + u_2 - u_3
        v_n = v_p - (1/lambda_2) * torch.transpose(torch.mm(f.proj(v_p,M),torch.transpose(f.gradient_v_2(u_p,v_p,Y,lambda_), 0, 1)), 0, 1) * dt + np.sqrt(2/(lambda_2*beta_v)) * torch.transpose(torch.mm(f.proj(v_p,M),torch.normal(0, np.sqrt(dt), size=(M, 1))),0,1) - ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        #print(i," : ", torch.linalg.norm(u_n), torch.linalg.norm(v_n))
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u_proj: ", res_u, "g2_v_proj: ", res_v)

    return (res_u, res_v)