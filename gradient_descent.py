import torch
import tensorflow as tf
import numpy as np
import util as f

# Number of iteration to perform the gradient descent
iteration = 100

# GRADIENT DESCENT WITH GRADIENT N°1

def main_gradient_1_sans_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        # Computation
        sqrt_dt = torch.sqrt(dt)

        u_1 = (1/lambda_1) * f.gradient_u_1(N, M, u_p,v_p,Y,lambda_) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.empty(N).normal_(mean=0,std=sqrt_dt)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3
        
        v_1 = 1/lambda_2 * f.gradient_v_1(N,M,u_p,v_p,Y,lambda_) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.empty(M).normal_(mean=0,std=sqrt_dt)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3
        
        # Re-asign for the loop
        u_p = u_n
        v_p = v_n

    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u: ", res_u)
    print("g1_v: ", res_v)

    return (res_u, res_v)

def main_gradient_1_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):
        # Computation
        sqrt_dt = torch.sqrt(dt)
        
        u_1 = (1/lambda_1) * f.gradient_u_1(N, M, u_p,v_p,Y,lambda_) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.empty(N).normal_(mean=0,std=sqrt_dt)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3
        
        v_1 = 1/lambda_2 * f.gradient_v_1(N,M,u_p,v_p,Y,lambda_) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.empty(M).normal_(mean=0,std=sqrt_dt)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3

        # Normalisation
        u_n = u_n / torch.linalg.norm(u_n)
        v_n = v_n / torch.linalg.norm(v_n)
        u_n = u_n * torch.sqrt(N)
        v_n = v_n * torch.sqrt(M)
        
        # Re-asign for the loop
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u_norm: ", res_u)
    print("g1_v_norm: ", res_v)

    return (res_u, res_v)

def main_gradient_1_avec_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        # Computation
        u_1 = (1/lambda_1) * torch.tensordot(f.proj(u_p,N), f.gradient_u_1(N, M, u_p,v_p,Y,lambda_), 1) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.tensordot(f.proj(u_p,N), torch.empty(N).normal_(mean=0,std=torch.sqrt(dt)), 1)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3

        v_1 = (1/lambda_2) * torch.tensordot(f.proj(v_p,M), f.gradient_v_1(N, M, u_p,v_p,Y,lambda_), 1) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.tensordot(f.proj(v_p,M), torch.empty(M).normal_(mean=0,std=torch.sqrt(dt)), 1)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3

        # Re-asign for the loop
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g1_u_proj: ", res_u)
    print("g1_v_proj: ", res_v)

    return (res_u, res_v)


# GRADIENT DESCENT WITH GRADIENT N°2

def main_gradient_2_sans_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        # Computation
        sqrt_dt = torch.sqrt(dt)

        u_1 = (1/lambda_1) * f.gradient_u_2(N, M, u_p,v_p,Y,lambda_) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.empty(N).normal_(mean=0,std=sqrt_dt)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3

        v_1 = 1/lambda_2 * f.gradient_v_2(N,M,u_p,v_p,Y,lambda_) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.empty(M).normal_(mean=0,std=sqrt_dt)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3

        # Re-asign for the loop
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u: ", res_u)
    print("g2_v: ", res_v)

    return (res_u, res_v)

def main_gradient_2_sans_proj_normalisation(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        # Computation
        sqrt_dt = torch.sqrt(dt)

        u_1 = (1/lambda_1) * f.gradient_u_2(N, M, u_p,v_p,Y,lambda_) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.empty(N).normal_(mean=0,std=sqrt_dt)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3

        v_1 = 1/lambda_2 * f.gradient_v_2(N,M,u_p,v_p,Y,lambda_) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.empty(M).normal_(mean=0,std=sqrt_dt)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3

        # Normalisation
        u_n = u_n / torch.linalg.norm(u_n)
        v_n = v_n / torch.linalg.norm(v_n)
        u_n = u_n * torch.sqrt(N)
        v_n = v_n * torch.sqrt(M)

        # Re-asign for the loop
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u_norm: ", res_u, "g2_v_norm: ", res_v)

    return (res_u, res_v)

def main_gradient_2_avec_proj(u_p, v_p, Y, u_, v_, N, M, lambda_, beta_u, beta_v, lambda_1, lambda_2, dt):

    for i in range(iteration):

        # Computation
        u_1 = (1/lambda_1) * torch.tensordot(f.proj(u_p,N), f.gradient_u_2(N, M, u_p,v_p,Y,lambda_), 1) * dt
        u_2 = torch.sqrt(2/(lambda_1*beta_u)) * torch.tensordot(f.proj(u_p,N), torch.empty(N).normal_(mean=0,std=torch.sqrt(dt)), 1)
        u_3 = ((N-1)/(N*lambda_1*beta_u))*u_p*dt
        u_n = u_p - u_1 + u_2 - u_3

        v_1 = (1/lambda_2) * torch.tensordot(f.proj(v_p,M), f.gradient_v_2(N, M, u_p,v_p,Y,lambda_), 1) * dt
        v_2 = torch.sqrt(2/(lambda_2*beta_v)) * torch.tensordot(f.proj(v_p,M), torch.empty(M).normal_(mean=0,std=torch.sqrt(dt)), 1)
        v_3 = ((M-1)/(M*lambda_2*beta_v))*v_p*dt
        v_n = v_p - v_1 + v_2 - v_3
        
        #print("# norm_u: ", torch.linalg.norm(u_p), " norm_v: ", torch.linalg.norm(v_p))

        # Re-asign for the loop
        u_p = u_n
        v_p = v_n
    
    res_u = f.overlap(u_,u_n,N)
    res_v = f.overlap(v_,v_n,M)
    print("g2_u_proj: ", res_u, "g2_v_proj: ", res_v)

    return (res_u, res_v)