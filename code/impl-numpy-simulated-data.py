# -*- coding: utf-8 -*-
"""
Numpy version of ST-Assign's implementation used to obtain resuls on simulated data
"""

import pandas as pd
import numpy as np
from scipy.stats import multinomial
from scipy.stats import gamma
from scipy.stats import norm
import scipy.special as sc
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import poisson
from numpy import random


import os


def run_setup(setup_number):
    setup_number = str("%02d" % setup_number)
    adres_parameters = "path to data/okolo5/setup" + setup_number  + "/"
    adres_results = "path to results/results/okolo5/setup" +  setup_number  + "/"
    os.makedirs(adres_results, exist_ok=True)

    TRUE_N = pd.read_csv(adres_parameters + "TRUE_N.csv", header=None).to_numpy().flatten()
    C_gs = pd.read_csv(adres_parameters + "C_gs.csv", index_col=0).to_numpy().astype("float")
    C_gc = pd.read_csv(adres_parameters + "C_gc.csv").to_numpy().astype("float")
    B = pd.read_csv(adres_parameters + "matB.csv", index_col=0).to_numpy()
    TRUE_rho = pd.read_csv(adres_parameters + "TRUE_rho.csv", header=None).to_numpy()
    TRUE_rho = TRUE_rho[:, 0] 
    TRUE_TC = pd.read_csv(adres_parameters + "TRUE_TC.csv", header=None).to_numpy()
    TRUE_TC = TRUE_TC[:,0].flatten()
    nTypes = B.shape[1]
    nSpots = C_gs.shape[1]
    nGenes = C_gs.shape[0]
    nCells = C_gc.shape[1]
    
    ile_po_kolei = B.sum(axis=1)
    prior_lambdas = np.apply_along_axis(lambda v: np.mean(v[v!=0]), 1, C_gs)
    lambdas_start = np.repeat(prior_lambdas, ile_po_kolei)

    a = 10
    b = 1
    a_0 = 0.1
    b_0 =1
    alpha = 0.5*nTypes

    
    pd.DataFrame().to_csv(adres_results + "est_M.csv", header=False, index=False)

    def my_logpmf_nb(x, n, p):
        coeff = sc.gammaln(n+x) - sc.gammaln(x+1) - sc.gammaln(n)
        return coeff + n*np.log(p) + sc.xlog1py(x, -p)
    
    def proposal_thetas(thetas_prev, step_size_thetas):
        return(my_trunc_norm_sampling_matrix(thetas_prev, step_size_thetas) )
    
    def my_trunc_norm_sampling_matrix(mu, sigma):
        U = np.random.mtrand._rand.uniform(size=(nTypes,nSpots))
        y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
        return y
    
    def log_prior_thetas(thetas, Z, aa, b, a_0, b_0):
        prior = gamma.logpdf(thetas, a=aa, scale=b)
        prior_epsilon = gamma.logpdf(thetas, a=a_0, scale=b_0)
        #print((Z==0).shape)
        prior[Z==0] = prior_epsilon[Z==0]
        return prior.sum(axis=1)
    
    def log_lik_thetas(C, M, thetas, N):
        row_sums = thetas.sum(axis=1)
        H = thetas/row_sums[:, np.newaxis]
        return  multinomial.logpmf(M, N, H)
    
    def update_thetas(C, M, curr_thetas, Z, Lambdas, N, step_size_thetas, a, b, a_0, b_0):
        prop_thetas = proposal_thetas(curr_thetas.T, step_size_thetas.T).T
        log_lik_prop = log_lik_thetas(C, M, prop_thetas, N)
        log_lik_curr = log_lik_thetas(C, M, curr_thetas, N)
            
        log_prior_prop = log_prior_thetas(prop_thetas, Z, a, b, a_0, b_0)
        log_prior_curr = log_prior_thetas(curr_thetas, Z, a, b, a_0, b_0)
        
        bias_prop = norm.logcdf(prop_thetas.T, loc=0, scale=step_size_thetas.T).sum(axis=0)
        bias_curr = norm.logcdf(curr_thetas.T, loc=0, scale=step_size_thetas.T).sum(axis=0)
        
        r = log_lik_prop -  log_lik_curr  + log_prior_prop - log_prior_curr  + bias_curr - bias_prop
        
        los = uniform.rvs(size=nSpots)
        decision = r> np.log(los)
        curr_thetas[decision] = prop_thetas[decision]
        return (curr_thetas, decision, r)
    
    def update_Z(current_thetas, current_pi, a, b, a_0, b_0):
        prob_0 = gamma.logpdf(current_thetas, a_0, scale=b_0) + np.log(1-current_pi)
        prob_1 = gamma.logpdf(current_thetas, a, scale=b) + np.log(current_pi)
        prob = np.exp(prob_0-prob_1)
        p = prob/(1+prob)
        
        #where_are_NaNs = np.isnan(p)
        #p[where_are_NaNs] = 0.5
        return   binom.rvs(1, 1- p)
    
    def my_trunc_norm_sampling_vector(mu, sigma):
        n = len(mu)
        U = np.random.mtrand._rand.uniform(size=n)
        y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
        return y
    
    
    def my_trunc_norm_sampling_lambda(mu, sigma):
        n = 1
        U = np.random.mtrand._rand.uniform(size=n)
        y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
        return y
    
    def proposal_n_cells(n_cells, step_size_n_cells):
        return my_trunc_norm_sampling_vector(n_cells, step_size_n_cells)
    
    def tNCDF(x, mu, sigma):
        return 1 - ( (1 - norm.cdf(  (x - mu )/sigma  ))/( norm.cdf(mu/sigma)))
    
    def density_ceil_tNorm(x, mu, sigma):
        return tNCDF(x, mu, sigma) - tNCDF(x-1, mu, sigma)
    
    def my_trunc_norm_sampling_0_1(mu, sigma):
        n = len(mu)
        U = np.random.mtrand._rand.uniform(size=n)
        y = mu + sigma*sc.ndtri(  U*(    sc.ndtr( (1-mu)/sigma  )  - sc.ndtr(-mu/sigma) )   +  sc.ndtr(-mu/sigma) )
        return y
    
    def proposal_p_g(curr_p_g, step_size_p_g):
    
        prop =  my_trunc_norm_sampling_0_1(curr_p_g, step_size_p_g)
        return prop
    
    def log_lik_p_g(C, M, Lambda, p_g):
        pg_factor =(1-p_g)/p_g
        p_g_m = np.tile(1-p_g,(nSpots,1) )
        mu_gs = np.matmul(Lambda, np.transpose(M))
        r_gs = ((mu_gs.T)*pg_factor).T
        return np.sum( my_logpmf_nb(C,r_gs,p_g_m.T), axis=1) 
    
    sum_row = lambda x: np.sum(x, axis=1)
    def log_lik_p_g_single_cell(C, TC, over_lambda, lambda_0, p_g, rho, rho_0):
        over_lambda = ((over_lambda.T)*rho).T
        Lambda = over_lambda +  lambda_0*rho_0
        pg_factor = (1-p_g)/p_g
        p_g_m = np.tile(1-p_g,(nCells,1) )
        r_gs = ((Lambda.T)*pg_factor).T
        r_gs = r_gs[:,TC-1]
        A = np.r_[np.array([TC]), my_logpmf_nb(C, r_gs, p_g_m.T)]
        A= A[:, A[0, :].argsort()]
        BB = np.split(A, np.unique(A[0, :], return_index=True)[1][1:], axis=1)
        temp = np.stack( list(map(sum_row, BB)), axis=0 ).T
        temp = np.delete(temp, (0), axis=0)
        return temp
    
    def helper_p_g(p_g, sigma):
        return  norm.cdf( (1 - p_g)/sigma  )  - norm.cdf( -p_g/sigma )
    
    
    def update_p_g(C, over_lambdas, lambda_0, curr_p_g, M, step_size_p_g, C_sc, TC, rho, rho_0):
        prop_p_g = proposal_p_g(curr_p_g ,step_size_p_g) 
        log_bias = np.log( helper_p_g(prop_p_g, step_size_p_g)) - np.log(helper_p_g(curr_p_g, step_size_p_g)  )
        r =  log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, prop_p_g, rho, rho_0).sum(axis=1) - log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, curr_p_g, rho, rho_0).sum(axis=1) + log_bias
        los = uniform.rvs(size=nGenes)
        decision = r> np.log(los)
        curr_p_g[decision] = prop_p_g[decision]
        return (curr_p_g, decision, r)
      
    def log_lik_M(M, C, Lambda, p):
        mu_gs = np.matmul(Lambda, np.transpose(M))
        return  np.sum(poisson.logpmf(C,mu_gs), axis=0)  
    
    
    def cal_p(M, S):
        p = (M.T + S).T
        row_sums = p.sum(axis=1)
        return p/row_sums[:, np.newaxis]
        
    
    vector_zeros = np.zeros(nSpots)
    def proposal_M_N(N, M, step_size_M, step_size_N):
        prop_N =  np.ceil(proposal_n_cells(N, step_size_N))
        S = my_trunc_norm_sampling_vector(vector_zeros, step_size_M )
        p_curr = cal_p(M, S)
        pN = np.column_stack((p_curr, prop_N))
        prop_M = np.apply_along_axis(lambda x: random.multinomial(x[nTypes], x[0:nTypes], size=None)  , 1, pN)
        return (prop_N, prop_M, S)
     
    
    def update_N_M(curr_N, curr_M, C, H, Lambda, p, step_size_N, step_size_M, prev_S, n_cells_prior):    
        
        prop =  proposal_M_N(curr_N, curr_M, step_size_M, step_size_N)
        prop_N = prop[0]
        prop_M = prop[1]
        S = prop[2]
        p_s_curr = cal_p(prop_M, prev_S)
        p_s_prop =  cal_p(curr_M, S)
          
        log_curr_lik = multinomial.logpmf(curr_M, curr_N, H) +  log_lik_M(curr_M, C, Lambda, p) + poisson.logpmf(curr_N, n_cells_prior)
        log_prop_lik = multinomial.logpmf(prop_M, prop_N, H) +  log_lik_M(prop_M, C, Lambda, p) + poisson.logpmf(prop_N, n_cells_prior)
           
        log_bias_curr = np.log(density_ceil_tNorm( curr_N, prop_N,   step_size_N)) + multinomial.logpmf(curr_M, curr_N, p_s_curr)
        log_bias_prop = np.log(density_ceil_tNorm( prop_N, curr_N, step_size_N)) + multinomial.logpmf(prop_M, prop_N, p_s_prop)
        r = log_prop_lik - log_curr_lik  +  log_bias_curr - log_bias_prop
        los = uniform.rvs(size=nSpots)
        decision = r > np.log(los)
    
        curr_N[decision] =  prop_N[decision] 
        curr_M[decision] =  prop_M[decision]  
          
        return (curr_N, curr_M, decision, r, S)
    
    
    def proposal_over_lambdas(over_lambdas, step_size):
        return(my_trunc_norm_sampling_vector(over_lambdas, step_size))
    
    def log_lik_over_lambdas(C, M, over_lambdas, lambda_0, p):
        Lambda = over_lambdas + lambda_0
        mu_gs = np.matmul(Lambda, np.transpose(M))
        return  np.nansum(poisson.logpmf(C, mu_gs), axis=1)   
    
    def proposal_rho(rho, step_size):
        return(my_trunc_norm_sampling_vector(rho, step_size))
    
    def update_rho(over_lambdas, lambda_0, p_g, step_size_rho, C_sc, TC, curr_rho, curr_rho_0, rho):
        prop_rho = proposal_rho(curr_rho, step_size_rho)
        log_curr_lik = log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, curr_rho_0).sum(axis=1) + norm.logpdf(curr_rho, loc=rho, scale=0.1)
        log_prop_lik =  log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, prop_rho, curr_rho_0).sum(axis=1) + norm.logpdf(prop_rho, loc=rho, scale=0.1)
        
        log_bias_prop = norm.logcdf(prop_rho, loc=0, scale=step_size_rho)
        log_bias_curr = norm.logcdf(curr_rho, loc=0, scale=step_size_rho)  
                
        r = log_prop_lik - log_curr_lik + log_bias_curr - log_bias_prop
        los = uniform.rvs(size=nGenes)
        decision = r> np.log(los)
        
        curr_rho[decision] = prop_rho[decision]
        return (curr_rho, decision, r)
    
    
    def update_rho_0(over_lambdas, lambda_0, p_g, step_size_rho_0, C_sc, TC, curr_rho, curr_rho_0):
        prop_rho_0 = proposal_lambda_0(curr_rho_0, step_size_rho_0)
        log_curr_lik = np.sum(log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, curr_rho_0)) + norm.logpdf(curr_rho_0, loc=1.5, scale=0.1)
        log_prop_lik =  np.sum(log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, prop_rho_0))  + norm.logpdf(prop_rho_0, loc=1.5, scale=0.1)
        
        log_bias_prop = np.sum(norm.logcdf(prop_rho_0, loc=0, scale=step_size_rho_0)) 
        log_bias_curr = np.sum(norm.logcdf(curr_rho_0, loc=0, scale=step_size_rho_0)) 
                
        r = log_prop_lik - log_curr_lik + log_bias_curr - log_bias_prop
    
        los = uniform.rvs(size=1)
        decision = r> np.log(los)
        
    
        if (decision):
            curr_rho_0 = prop_rho_0
        return (curr_rho_0, decision, r)
    
    wszystkie = np.arange(0, nTypes, 1, dtype=int)+1
    def log_lik_over_lambdas_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, rho, rho_0):
    
        Lambda =  ( (over_lambdas.T) * rho).T + lambda_0*rho_0
        pg_factor = (1-p_g)/p_g
        p_g_m = np.tile(1-p_g,(nCells,1) )
        r_gs = ((Lambda.T)*pg_factor).T
        r_gs = r_gs[:,TC-1]
        A = np.r_[np.array([TC]), my_logpmf_nb(C_sc, r_gs, p_g_m.T)]
        A= A[:, A[0, :].argsort()]
        BB = np.split(A, np.unique(A[0, :], return_index=True)[1][1:], axis=1)
        temp = np.stack( list(map(sum_row, BB)), axis=0 ).T
        temp = np.delete(temp, (0), axis=0)
        
        if (len(np.unique(TC)) != nTypes):
            roznica = np.setdiff1d(wszystkie, np.unique(np.array([TC])))
            xx = np.concatenate([np.unique(np.array([TC])), roznica])
            temp = np.concatenate([temp, np.zeros((nGenes, len(roznica)))], axis=1)
            temp = temp[:, np.argsort(xx)]
            
        return temp
    
    def update_over_lambdas(C, B, M, curr_over_lambdas, lambda_0, d, step_size, xx0, yy0, xx1, yy1, ind, how_many_each_type, C_sc, TC, rho, rho_0):
        vector_curr_over_lambdas = curr_over_lambdas[ind]
        step_size_over_lambdas = np.repeat(step_size,  how_many_each_type)
    
        vector_proposal_over_lambdas = proposal_over_lambdas(vector_curr_over_lambdas, step_size_over_lambdas)
        prop_over_lambdas = np.zeros((nGenes,nTypes))
        prop_over_lambdas[xx1,yy1] = vector_proposal_over_lambdas
        
        
        bias_prop = np.zeros((nGenes,nTypes), dtype=float)
        bias_curr = np.zeros((nGenes,nTypes), dtype=float)
        
    
        bias_prop[xx1, yy1] = norm.logcdf(vector_proposal_over_lambdas, loc=0, scale=step_size)
        bias_curr[xx1, yy1] = norm.logcdf(vector_curr_over_lambdas, loc=0, scale=step_size)  
        
        
        log_curr_lik = (log_lik_over_lambdas(C, M, curr_over_lambdas, lambda_0, d)  + log_lik_over_lambdas_single_cell(C_sc, TC, curr_over_lambdas, lambda_0, d, rho, rho_0).T).T
        log_prop_lik = (log_lik_over_lambdas(C, M, prop_over_lambdas, lambda_0, d) + log_lik_over_lambdas_single_cell(C_sc, TC,  prop_over_lambdas, lambda_0, d, rho, rho_0).T).T
        
        r = log_prop_lik.sum(axis=1) - log_curr_lik.sum(axis=1)  + bias_curr.sum(axis=1) - bias_prop.sum(axis=1) 
    
        los = uniform.rvs(size=nGenes)
        decision = np.where(r> np.log(los))
        curr_over_lambdas[decision,] = prop_over_lambdas[decision,]
        vector_curr_over_lambdas = curr_over_lambdas[ind]
        
        return (curr_over_lambdas, decision, r, vector_curr_over_lambdas) 
    
    
    def proposal_lambda_0(curr_lambda_0, step_size_lambda_0):
        return my_trunc_norm_sampling_lambda(curr_lambda_0, step_size_lambda_0)
    
    def log_lik_lambda_0_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, rho, rho_0):
        Lambda =  ((over_lambdas.T)*rho).T + lambda_0*rho_0
        pg_factor = (1-p_g)/p_g
        p_g_m = np.tile(1-p_g,(nCells,1) )
        r_gs = ((Lambda.T)*pg_factor).T
        r_gs = r_gs[:,TC-1]
        A = np.r_[np.array([TC]), my_logpmf_nb(C_sc, r_gs, p_g_m.T)]
        A= A[:, A[0, :].argsort()]
        BB = np.split(A, np.unique(A[0, :], return_index=True)[1][1:], axis=1)
        temp = np.stack( list(map(sum_row, BB)), axis=0 ).T
        temp = np.delete(temp, (0), axis=0)
        return np.sum(temp)
    
    def log_lik_lambda_0(C, M, lambda_0, over_lambda, pg, C_sc, TC, rho, rho_0):
        Lambda = lambda_0 + over_lambda
        mu_gs =  np.matmul(Lambda, np.transpose(M))
        res =  log_lik_lambda_0_single_cell(C_sc, TC, over_lambda, lambda_0, pg, rho, rho_0) + np.sum(poisson.logpmf(C, mu_gs))
        return  res
    
    def update_lambda_0(C, M, curr_lambda_0, over_lambda, d, step_size_lambda_0, C_sc, TC, rho, rho_0):
        prop_lambda_0 = proposal_lambda_0(curr_lambda_0, step_size_lambda_0)
        
        log_prop_lik = log_lik_lambda_0(C, M, prop_lambda_0, over_lambda, d, C_sc, TC, rho, rho_0)
        log_curr_lik = log_lik_lambda_0(C, M, curr_lambda_0, over_lambda, d, C_sc, TC, rho, rho_0)
        
        bias_prop = norm.logcdf(prop_lambda_0, loc=0, scale=step_size_lambda_0)
        bias_curr = norm.logcdf(curr_lambda_0, loc=0, scale=step_size_lambda_0)
        
        r = log_prop_lik - log_curr_lik  + bias_curr - bias_prop 
        los = uniform.rvs(size=1)
        decision = r> np.log(los)
    
        if (decision):
            curr_lambda_0 = prop_lambda_0
        return (curr_lambda_0, decision, r) 
    
    
    def random_choice_prob_index(a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)
    
    def log_lik_TC_single_type(C_sc, TC, over_lambdas, lambda_0, p_g, rho, rho_0):
        Lambda = ((over_lambdas.T)*rho).T + lambda_0*rho_0
        p_g_m = np.tile(1-p_g,(nCells,1) )
        pg_factor = (1-p_g)/p_g
        p_g_m = np.tile(1-p_g,(nCells,1) )
        r_gs = ((Lambda.T)*pg_factor).T
        r_gs = np.tile(np.array(r_gs[:,TC-1]), (nCells,1)).T
        return np.sum(my_logpmf_nb(C_sc, r_gs, p_g_m.T), axis=0)
    
    
    def update_TC(C_sc, over_lambdas, lambda_0, p_g, rho, rho_0, D):
        final = []
        for ii in range(0, nTypes):
            final.append(   log_lik_TC_single_type(C_sc, ii+1, over_lambdas, lambda_0, p_g, rho, rho_0)  )
            
        A = np.array(final)
        A = A.T    + np.log(D)
        v = - np.log( - np.log(  np.random.uniform(0,1,(nCells,nTypes))))
        res = np.argmax(v + A, axis=1)+1
        return res
    
    
    def update_D(TC, d):
        TC= np.append(TC, np.arange(1,nTypes+1))
        (unique, counts) = np.unique(TC, return_counts=True)
        d_prim = d + counts-1
        
        return random.dirichlet(d_prim)
    

    def Gibbs():
        
        burn_in = 80000
        number_of_it = 100000
        licznik = 0
        sum_M = np.zeros((nSpots,nTypes))
        
        current_lambda_0 = uniform.rvs(loc=0, scale=2, size=1)
        current_pi = 0.5*np.ones((nSpots,nTypes))
        current_Z = np.ones((nSpots,nTypes)).astype(int)
        current_thetas =  gamma.rvs(a=a*np.ones((nSpots,nTypes)), scale=b)
    
        d = (1/nTypes)*np.ones((nTypes))
        current_D = (1/nTypes)*np.ones(nTypes)   
         
        ind_x0, ind_y0= np.where(current_Z == 0)
        current_thetas[ind_x0, ind_y0] = gamma.rvs(a=a_0*np.ones(  (1-current_Z).sum() ), scale=b_0)
        current_N =  poisson.rvs(size=nSpots, mu=4)+1
        current_pg =  uniform.rvs(loc=0, scale=1, size=nGenes, random_state=None)
     
        current_rho = np.ones(nGenes)
        prob = np.tile(np.repeat(1/nTypes, nTypes), (nCells, 1))
        current_TC = random_choice_prob_index(prob)+1
        pd.DataFrame(current_TC).T.to_csv(adres_results + "res_TC.csv", header=False, index=False, mode="a")

        current_rho_0 = np.ones(1)
              
        current_M = np.zeros((nSpots,nTypes))
        vec = np.repeat(1/nTypes,nTypes)
        
        for i in range(0,nSpots):
             current_M[i,:] = multinomial.rvs(current_N[i], vec, size=1)
        
        vec = np.repeat(1/nTypes,nTypes)
        func = lambda t:  multinomial.rvs(t, vec, size=1 )
        current_M = np.array(list(map(func, current_N)))[:, 0, :]
            
        step_size_thetas =  0.1*np.ones((nSpots,nTypes))
        step_size_N = 2.5*np.ones(nSpots)
        step_size_M = 2.5*np.ones( nSpots )   
        step_size_pg = 0.5*np.ones(nGenes)
        step_size_over_lambdas = 4*np.ones(nGenes)
        step_size_rho_0 = 1
        step_size_over_lambdas = prior_lambdas/5
        out = np.where(step_size_over_lambdas<2)
        step_size_over_lambdas[out] = 2
        
        step_size_lambda_0 = 1
        step_sized_rho =5*np.ones(nGenes)
    
        xx1, yy1 = np.where(B == 1)   
        xx0, yy0 = np.where(B == 0)
        how_many_each_type = B.sum(axis=1)
        ind = np.array(B, dtype=bool)
    
        current_over_lambdas = np.zeros((nGenes, nTypes), dtype=float)
        current_over_lambdas[xx1,yy1] = lambdas_start/10
        prev_S = 0
    
        
        for i in range(0, number_of_it):
            print(i)
        
            current_pi = beta.rvs( alpha/nTypes + current_Z, 2-current_Z)
            current_Z = update_Z(current_thetas, current_pi, a, b, a_0, b_0)
            current_Z[:, nTypes-1] = 0
            
            res_thetas = update_thetas(C_gs, current_M, current_thetas, current_Z, current_over_lambdas + current_lambda_0, current_N, step_size_thetas, a, b, a_0, b_0)
            current_thetas = res_thetas[0]

            row_sums = current_thetas.sum(axis=1)
            curr_H = current_thetas/row_sums[:, np.newaxis]
            
            res_N_M = update_N_M(current_N, current_M, C_gs, curr_H, current_over_lambdas + current_lambda_0, current_pg, step_size_N,  step_size_M, prev_S, TRUE_N)
            current_M = res_N_M[1]
            current_N = res_N_M[0]

            prev_S =  res_N_M[4]
            
            res_over_lambdas = update_over_lambdas(C_gs, B, current_M, current_over_lambdas, current_lambda_0, current_pg, step_size_over_lambdas, xx0, yy0, xx1, yy1, ind, how_many_each_type, C_gc, current_TC, current_rho, current_rho_0)
            current_over_lambdas = res_over_lambdas[0]

            res_lambda_0 = update_lambda_0(C_gs, current_M, current_lambda_0, current_over_lambdas, current_pg, step_size_lambda_0, C_gc, current_TC, current_rho, current_rho_0)
            current_lambda_0 = res_lambda_0[0] 

            res_pg = update_p_g(C_gs, current_over_lambdas, current_lambda_0, current_pg,  current_M, step_size_pg, C_gc, current_TC, current_rho, current_rho_0)
            current_pg = res_pg[0]
    
            current_TC = update_TC(C_gc, current_over_lambdas, current_lambda_0, current_pg, current_rho, current_rho_0, current_D)
            
            current_D = update_D(current_TC, d)
            
            res_rho = update_rho(current_over_lambdas, current_lambda_0, current_pg, step_sized_rho, C_gc, current_TC, current_rho, current_rho_0, TRUE_rho)
            current_rho = res_rho[0]
            
            res_rho_0 = update_rho_0(current_over_lambdas, current_lambda_0, current_pg, step_size_rho_0, C_gc, current_TC, current_rho, current_rho_0)
            current_rho_0 = res_rho_0[0]
        

            if i>burn_in:
                licznik = licznik + 1
                sum_M = sum_M + current_M
                

            
        pd.DataFrame(sum_M/licznik).to_csv(adres_results + "est_M.csv", header=False, index=False)
        pd.DataFrame(current_TC).T.to_csv(adres_results + "res_TC.csv", header=False, index=False)
            
        
    Gibbs()
    
    
from joblib import Parallel, delayed
result = Parallel(n_jobs=10)(delayed(run_setup)(i) for i in range(1,11))

    
