# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.stats import multinomial
from scipy.stats import gamma
from scipy.stats import norm
import scipy.special as sc
from scipy.stats import uniform
from scipy.stats import poisson
from numpy import random


import os
import sys
import json


import tensorflow as tf
import tensorflow_probability as tfp



address =  sys.argv[1]
address_results =  sys.argv[2]

os.environ["CUDA_VISIBLE_DEVICES"]="0"     
with tf.device('/device:GPU:0'):
        

    with open(address + '/params.txt') as f:
        dd = f.read()
    params = json.loads(dd) 
    
    
    if not os.path.exists(address_results):
        os.makedirs(address_results)
 

    #C_gs = pd.read_csv(address + "/C_gs.csv", index_col=0).to_numpy()
    #tf_C_gs = tf.convert_to_tensor(C_gs, dtype="float32")
    #C_gc = pd.read_csv(address + "/C_gc.csv",  index_col=0).to_numpy()
    #tf_C_gc = tf.convert_to_tensor(C_gc, dtype="float32")
    #B = pd.read_csv(address + "/matB.csv", index_col=0).to_numpy()
    #tf_B = tf.convert_to_tensor(B, dtype="float32")
    
    C_gs = pd.read_csv(address + "/C_gs.csv", index_col=0)
    spoty = C_gs.columns
    geny_C_gs = C_gs.index
    C_gs = C_gs.to_numpy()
    tf_C_gs = tf.convert_to_tensor(C_gs, dtype="float32")

    C_gc = pd.read_csv(address + "/C_gc.csv",  index_col=0)
    geny_C_gc = C_gc.index
    C_gc = C_gc.to_numpy()
    tf_C_gc = tf.convert_to_tensor(C_gc, dtype="float32")

    B = pd.read_csv(address + "/matB.csv", index_col=0)
    geny_B = B.index
    B = B.to_numpy()
    tf_B = tf.convert_to_tensor(B, dtype="float32")
    
   
    
    rho = tf.convert_to_tensor( pd.read_csv(address + "/rho.csv").to_numpy().flatten(),   dtype="float32")
    
    nTypes = B.shape[1]
    nSpots = C_gs.shape[1]
    nGenes = C_gs.shape[0]
    nCells = C_gc.shape[1]
    
    
    ile_po_kolei = B.sum(axis=1)
    prior_lambdas = np.apply_along_axis(lambda v: np.mean(v[v!=0]), 1, C_gs)
    lambdas_start = np.repeat(prior_lambdas, ile_po_kolei)

   
    a = params['a']
    b = params['b']
    a_0 = params['a_0']
    b_0 = params['b_0']
    alpha = params['alpha']
    burn_in =  params['burn in']
    number_of_it = params['number of iterations']
    st_thetas = params['step size thetas']
    st_lambda0 = params['step size lambda0']
    st_M = params['step size M']
    st_N = params['step size N']
    st_pg = params['step size pg']
    st_rho = params['step size rho']
    st_rho_0 = params['step size rho0']
    prior_rho_0 = params['prior rho0']
    number_of_cells = params['number of cells']
    

    if number_of_cells == 'estimated':
        n_cells =  tf.convert_to_tensor(pd.read_csv(address + "/n_cells.csv")["cellCount"].to_numpy(), dtype="float32")
        spotId = pd.read_csv(address + "/n_cells.csv")["spotId"]
        assert len(n_cells)==C_gs.shape[1], "the number of spots in n_cells.csv and C_gs.csv should be the same"
        assert all(spotId.to_numpy()==spoty.to_numpy()), "the order of spots in n_cells.csv and C_gs.csv should be the same"
    else:
        n_cells = tf.cast(np.tile( float(number_of_cells), nSpots), tf.float32)
        
    assert np.all((B==0) | (B==1)), "elements of B should be 0 or 1"
    #assert len(n_cells.shape)==1, "number of cells should be one dimentional vector"
    #n_cells_int = n_cells.astype(int)
    #assert np.all(n_cells - n_cells_int ==0), "number of cells should be integers"
    C_gc_int = C_gc.astype(int)
    C_gs_int = C_gs.astype(int)
    assert np.all(C_gc - C_gc_int ==0), "gene expression counts should be integers"
    assert np.all(C_gs - C_gs_int ==0), "gene expression counts should be integers"
    assert C_gs.shape[0] == B.shape[0], "number of rows of C_gc should be equal to number of rows of B"
    assert C_gc.shape[0] == B.shape[0], "number of rows of C_gs should be equal to number of rows of B"
    assert burn_in < number_of_it, "burn-in should be smaller than number of iterations"
    assert C_gs.shape[0] == len(number_of_cells)
    assert all(geny_C_gs==geny_B), "the order of genes in C_gs should match the order in B"
    assert all(geny_C_gc==geny_B), "the order of genes in C_gc should match the order in B"
    

    pd.DataFrame().to_csv(address_results + "/res_M.csv", header=False, index=False)
    pd.DataFrame().to_csv(address_results +"/res_TC.csv", header=False, index=False)
    
    
   
    def proposal_thetas(thetas_prev, step_size_thetas):
        return(my_trunc_norm_sampling_matrix(thetas_prev, step_size_thetas) )
    
    def my_trunc_norm_sampling_matrix(mu, sigma):
        U = np.random.mtrand._rand.uniform(size=(nTypes,nSpots))
        y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
        return y
    
    def tf_my_trunc_norm_sampling_matrix_thetas(mu, sigma, nTypes, nSpots):
        U = tf.random.uniform(shape=(nTypes, nSpots))
        ndtr_minus = tfp.distributions.Normal(0, 1).cdf(-mu / sigma)
        y = mu + sigma * tfp.distributions.Normal(0, 1).quantile(U + ndtr_minus * (1 - U))
        return tf.cast( y, "float32")
    
    def tf_my_trunc_norm_sampling_lambda(mu, sigma):
        U =tf.cast(uniform.rvs(size=1), dtype="float32")
        ndtr_minus = tfp.distributions.Normal(0, 1).cdf(-mu / sigma)
        y = mu + sigma * tfp.distributions.Normal(0, 1).quantile(U + ndtr_minus * (1 - U))
        return tf.cast( y, "float32")
    
    def tf_proposal_thetas(thetas_prev, step_size_thetas, nTypes, nSpots):
        return tf_my_trunc_norm_sampling_matrix_thetas(thetas_prev, step_size_thetas, nTypes, nSpots)
    
    def tf_log_prior_thetas(thetas, Z, aa, b, a_0, b_0):
        prior = tfp.distributions.Gamma(aa, b).log_prob(thetas)
        prior_epsilon = tfp.distributions.Gamma(a_0, b_0).log_prob(thetas)
        prior = tf.where(Z == 0, prior_epsilon, prior)
        return tf.cast(tf.reduce_sum(prior, axis=1),  tf.float32)
    
    def tf_log_lik_thetas(M, thetas, N):
        row_sums = tf.reduce_sum(thetas, axis=1)
        H = thetas / tf.expand_dims(row_sums, axis=1)
        multinomial = tfp.distributions.Multinomial(total_count=N, probs=H)
        log_likelihood = multinomial.log_prob(M)
        return tf.convert_to_tensor(log_likelihood, dtype=tf.float32)
    
    
    def tf_update_thetas(M, curr_thetas, Z, Lambdas, N, step_size_thetas, a, b, a_0, b_0, nTypes, nSpots):
        prop_thetas = tf.transpose(tf_proposal_thetas(tf.transpose(curr_thetas), tf.transpose(step_size_thetas), nTypes, nSpots))
        log_lik_prop = tf_log_lik_thetas(M, prop_thetas, N)
        log_lik_curr = tf_log_lik_thetas(M, curr_thetas, N)
        log_prior_prop = tf_log_prior_thetas(prop_thetas, Z, a, b, a_0, b_0)
        log_prior_curr = tf_log_prior_thetas(curr_thetas, Z, a, b, a_0, b_0)
        bias_prop = tf.cast(tf.reduce_sum(tfp.distributions.Normal(0, step_size_thetas).log_cdf(prop_thetas),1), tf.float32)
        bias_curr = tf.cast(tf.reduce_sum(tfp.distributions.Normal(0, step_size_thetas).log_cdf(curr_thetas),1), tf.float32)
        r = log_lik_prop - log_lik_curr + log_prior_prop - log_prior_curr + bias_curr - bias_prop
        los = tf.random.uniform(shape=(nSpots,))
        decision = r > tf.math.log(los)
        curr_thetas = tf.where(decision[:, tf.newaxis], prop_thetas, curr_thetas)
        return curr_thetas
    
    def tf_update_Z(current_thetas, current_pi, a, b, a_0, b_0):
        prob_0 = tfp.distributions.Gamma(a_0, b_0).log_prob(current_thetas) + tf.math.log(1 - current_pi)
        prob_1 = tfp.distributions.Gamma(a, b).log_prob(current_thetas) + tf.math.log(current_pi)
        prob = tf.exp(prob_0 - prob_1)
        p = prob / (1 + prob)
        return tfp.distributions.Binomial(1, 1 - p).sample()
    
    def my_trunc_norm_sampling_vector(mu, sigma):
        n = len(mu)
        U = np.random.mtrand._rand.uniform(size=n)
        y = mu + sigma*sc.ndtri(U+sc.ndtr(-mu/sigma)*(1-U))
        return y

    def proposal_n_cells(n_cells, step_size_n_cells):
        return my_trunc_norm_sampling_vector(n_cells, step_size_n_cells)
    
    def tNCDF(x, mu, sigma):
        return 1 - ( (1 - norm.cdf(  (x - mu )/sigma  ))/( norm.cdf(mu/sigma)))

    def tf_tNCDF(x, mu, sigma):
        dist = tfp.distributions.Normal(0,1)
        norm_cdf_x = dist.cdf((x - mu) / sigma)
        norm_cdf_mu = dist.cdf(mu / sigma)
        return tf.cast(1 - ((1 - norm_cdf_x) / norm_cdf_mu), "float32")
    
    def tf_density_ceil_tNorm(x, mu, sigma):
        return tf.cast(tNCDF(x, mu, sigma) - tNCDF(x - 1, mu, sigma), "float32")
    
    def tf_my_trunc_norm_sampling_vector(mu, sigma):
        n = tf.shape(mu)[0]
        U = tf.random.uniform(shape=(n,))
        y = mu + sigma * (tfp.distributions.Normal(0, 1).quantile(U + tfp.distributions.Normal(0, 1).cdf(-mu / sigma) * (1 - U)))
        return y
    
    def tf_proposal_n_cells(n_cells, step_size_n_cells):
         return my_trunc_norm_sampling_vector(n_cells, step_size_n_cells)
     
    vector_zeros = tf.zeros([nSpots], tf.float32)
    def tf_proposal_M_N(N, M, step_size_M, step_size_N):
        prop_N =  tf.math.ceil(tf_proposal_n_cells(N, step_size_N))
        S = tf_my_trunc_norm_sampling_vector(vector_zeros, step_size_M )
        p_curr = tf_cal_p(M, S)
        dist = tfp.distributions.Multinomial(total_count=prop_N, probs=p_curr)
        prop_M = dist.sample()
        return (prop_N, prop_M, S)
    
    def my_trunc_norm_sampling_0_1(mu, sigma):
        n = len(mu)
        U = np.random.mtrand._rand.uniform(size=n)
        y = mu + sigma*sc.ndtri(  U*(    sc.ndtr( (1-mu)/sigma  )  - sc.ndtr(-mu/sigma) )   +  sc.ndtr(-mu/sigma) )
        return y
    
    def tf_my_trunc_norm_sampling_0_1(mu, sigma):
        n = len(mu)
        U = tf.random.uniform(shape=(n,))
        dist = tfp.distributions.Normal(0, 1)
        y = mu + sigma*dist.quantile(  U*(dist.cdf((1-mu)/sigma)  - dist.cdf(-mu/sigma)) + dist.cdf(-mu/sigma))
        return y
    
    def tf_proposal_p_g(curr_p_g, step_size_p_g):
        prop =  tf_my_trunc_norm_sampling_0_1(curr_p_g, step_size_p_g)
        return prop
    
    def tf_log_lik_p_g(C, M, Lambda, p_g, nSpots):
        pg_factor = (1 - p_g) / p_g
        p_g_m = tf.tile(tf.expand_dims(p_g, 0), (nSpots,1))
        mu_gs = tf.matmul(Lambda, tf.transpose(M))
        r_gs = tf.transpose(tf.transpose(mu_gs) * pg_factor)
        dist = tfp.distributions.NegativeBinomial(total_count=r_gs, probs=tf.transpose(p_g_m))
        return tf.reduce_sum(dist.log_prob(C), axis=1)
    
    def tf_log_lik_p_g_single_cell(C_sc, TC, over_lambda, lambda_0, p_g, rho, rho_0, nCells):
        over_lambda = tf.transpose(tf.transpose(over_lambda) * rho)
        Lambda_ = over_lambda + lambda_0 * rho_0
        pg_factor = (1 - p_g) / p_g
        p_g_m = tf.tile(tf.expand_dims(p_g, 0), [nCells,1])
        r_gs = tf.transpose(tf.transpose(Lambda_) * pg_factor)
        r_gs = tf.gather(r_gs, TC-1, axis=1)
        dist = tfp.distributions.NegativeBinomial(total_count=r_gs, probs=tf.transpose(p_g_m))
        A = tf.concat([ tf.cast(tf.reshape(TC, (1, -1)), dtype="float32"), dist.log_prob(C_sc)], axis=0)
        sorted_indices = tf.argsort(A[0, :])
        A = tf.gather(A, sorted_indices, axis=1)
        unique_elements, _ = tf.unique(A[0, :])
        BB = []
        for unique_element in unique_elements:
            mask = tf.equal(A[0, :], unique_element)
            split = tf.boolean_mask(A, mask, axis=1)
            BB.append(split)
            
        sums = []
        for tensor in BB:
            sum_result = tf.reduce_sum(tensor, axis=1, keepdims=True)
            sums.append(sum_result)
    
        temp = tf.concat(sums, axis=1)
        temp = tf.slice(temp, [1, 0], [-1, -1])
        return temp
    
    def tf_helper_p_g(p_g, sigma):
        normal = tfp.distributions.Normal(loc=0.0, scale=1.0)  
        cdf1 = normal.cdf((1 - p_g) / sigma)
        cdf2 = normal.cdf(-p_g / sigma)
        return cdf1 - cdf2
    
    def tf_update_p_g(C, over_lambdas, lambda_0, curr_p_g, M, step_size_p_g, C_sc, TC, rho, rho_0, nSpots, nCells):
        prop_p_g = tf_proposal_p_g(curr_p_g ,step_size_p_g) 
        log_bias = tf.math.log( tf_helper_p_g(prop_p_g, step_size_p_g)) - tf.math.log(tf_helper_p_g(curr_p_g, step_size_p_g)  )
        r = tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, prop_p_g, rho, rho_0, nCells), axis=1)  - tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, curr_p_g, rho, rho_0, nCells), axis=1)  + log_bias
        los = tf.random.uniform(shape=[nGenes])
        decision = r > tf.math.log(los)
        curr_p_g = tf.where(decision, prop_p_g, curr_p_g)
        return (curr_p_g, decision, r)
    
    def tf_log_lik_M(M, C, Lambda, p):
        mu_gs = tf.matmul(Lambda, tf.transpose(M))
        dist = tfp.distributions.Poisson(rate=mu_gs)
        return tf.reduce_sum(dist.log_prob(C), axis=0)
    
    def tf_cal_p(M, S):
        p = tf.transpose(tf.transpose(M) + S)
        row_sums = tf.reduce_sum(p, axis=1, keepdims=True)
        return p / row_sums
        
    def tf_update_N_M(curr_N, curr_M, C, H, Lambda, p, step_size_N, step_size_M, prev_S, n_cells_prior):    
        prop =  tf_proposal_M_N(curr_N, curr_M, step_size_M, step_size_N)
        prop_N = prop[0]
        prop_M = prop[1]
        S = prop[2]
        p_s_curr = tf_cal_p(prop_M, prev_S)
        p_s_prop =  tf_cal_p(curr_M, S)
          
        dist_curr = tfp.distributions.Multinomial(total_count=curr_N, probs=H)
        dist_prop = tfp.distributions.Multinomial(total_count=prop_N, probs=H)
        dist_poisson= tfp.distributions.Poisson(n_cells_prior)
    
        log_curr_lik = dist_curr.log_prob(curr_M) + tf_log_lik_M(curr_M, C, Lambda, p) + dist_poisson.log_prob(curr_N)
        log_prop_lik = dist_prop.log_prob(prop_M) + tf_log_lik_M(prop_M, C, Lambda, p) + dist_poisson.log_prob(prop_N)
        
        dist_curr = tfp.distributions.Multinomial(total_count=curr_N, probs=p_s_curr)
        dist_prop = tfp.distributions.Multinomial(total_count=prop_N, probs=p_s_prop)
        log_bias_curr = tf.math.log(tf_density_ceil_tNorm( curr_N, prop_N,   step_size_N)) + dist_curr.log_prob(curr_M)
        log_bias_prop = tf.math.log(tf_density_ceil_tNorm( prop_N, curr_N, step_size_N)) + dist_prop.log_prob(prop_M)
        
        r = log_prop_lik - log_curr_lik  +  log_bias_curr - log_bias_prop
    
    
        los = tf.random.uniform(shape=[nSpots])
    
        decision = r > tf.math.log(los)
        curr_N = tf.where(decision, prop_N, curr_N)
        curr_M = tf.where(decision[:, tf.newaxis], prop_M, curr_M)
        return (curr_N, curr_M, decision, r, S)

    def proposal_over_lambdas(over_lambdas, step_size):
        return(my_trunc_norm_sampling_vector(over_lambdas, step_size))
    
    def tf_proposal_over_lambdas(over_lambdas, step_size):
        return(tf_my_trunc_norm_sampling_vector(over_lambdas, step_size))

    def tf_log_lik_over_lambdas(C, M, over_lambdas, lambda_0, p, nSpots):
        Lambda = over_lambdas + lambda_0
        mu_gs = tf.matmul(Lambda, tf.transpose(M))
        dist = tfp.distributions.Poisson(rate = mu_gs)
        return  tf.reduce_sum(dist.log_prob(C), axis=1)   
    
    def proposal_rho(rho, step_size):
        return(my_trunc_norm_sampling_vector(rho, step_size))
    
    wszystkie = tf.convert_to_tensor(np.arange(0, nTypes, 1, dtype=int)+1, dtype="int32")
    def tf_log_lik_over_lambdas_single_cell(C, TC, over_lambdas, lambda_0, p_g, rho, rho_0):
        Lambda_ = tf.transpose(tf.transpose(over_lambdas) * rho) + lambda_0 * rho_0
        pg_factor = (1 - p_g) / p_g
        p_g_m = tf.tile(tf.expand_dims(p_g, 0), [nCells,1])
        r_gs = tf.transpose(tf.transpose(Lambda_) * pg_factor)
        r_gs = tf.gather(r_gs, TC-1, axis=1)
        dist = tfp.distributions.NegativeBinomial(total_count=r_gs, probs=tf.transpose(p_g_m))
        
       
        A = tf.concat([ tf.cast(tf.reshape(TC, (1, -1)), dtype="float32"), dist.log_prob(C)], axis=0)
        sorted_indices = tf.argsort(A[0, :])
        A = tf.gather(A, sorted_indices, axis=1)
        unique_elements, _ = tf.unique(A[0, :])
        BB = []
    
        for unique_element in unique_elements:
            mask = tf.equal(A[0, :], unique_element)
            split = tf.boolean_mask(A, mask, axis=1)
            BB.append(split)
            
        sums = []
        for tensor in BB:
            sum_result = tf.reduce_sum(tensor, axis=1, keepdims=True)
            sums.append(sum_result)
    
        temp = tf.concat(sums, axis=1)
        temp = tf.slice(temp, [1, 0], [-1, -1])
        
        
        unikalne = tf.cast(tf.unique(TC)[0], "int32")
    
        roznica = tf.sets.difference(tf.expand_dims(wszystkie, axis=0), tf.expand_dims(unikalne, axis=0)).values
        xx = tf.concat([unikalne, roznica], axis=0)
        zeros_shape = (nGenes, tf.shape(roznica)[0])
        zeros = tf.zeros(zeros_shape, dtype=tf.float32)
    
        temp = tf.concat([temp, zeros], axis=1)
        sorted_indices = tf.argsort(xx)
        temp = tf.gather(temp, sorted_indices, axis=1)
        return temp
    
    def tf_update_over_lambdas(C, B, M, curr_over_lambdas, lambda_0, d, step_size, xx0, yy0, xx1, yy1, ind, how_many_each_type, C_sc, TC, rho, rho_0, indices_ones, indices_zeros):
        vector_curr_over_lambdas = tf.boolean_mask(curr_over_lambdas, ind)
        step_size_over_lambdas = np.repeat(step_size,  how_many_each_type)
        
        vector_proposal_over_lambdas = tf_proposal_over_lambdas(vector_curr_over_lambdas, step_size_over_lambdas)
        prop_over_lambdas = tf.zeros((nGenes, nTypes), dtype=tf.float32)
        prop_over_lambdas = tf.tensor_scatter_nd_update(prop_over_lambdas, indices_ones, vector_proposal_over_lambdas)
        
        bias_prop = tf.zeros((nGenes, nTypes), dtype=tf.float32)
        bias_curr = tf.zeros((nGenes, nTypes), dtype=tf.float32)
        
        normal_dist = tfp.distributions.Normal(loc=0.0, scale=step_size_over_lambdas)
    
        logcdf_prop = normal_dist.log_cdf(vector_proposal_over_lambdas)
        logcdf_curr = normal_dist.log_cdf(vector_curr_over_lambdas)
    
        bias_prop = tf.tensor_scatter_nd_update(bias_prop, indices_ones, logcdf_prop)
        bias_curr = tf.tensor_scatter_nd_update(bias_curr, indices_ones, logcdf_curr)
    
        log_curr_lik = tf.transpose(tf_log_lik_over_lambdas(C, M, curr_over_lambdas, lambda_0, d, nSpots)  + tf.transpose(tf_log_lik_over_lambdas_single_cell(C_sc, TC, curr_over_lambdas, lambda_0, d, rho, rho_0)))
        log_prop_lik =  tf.transpose(tf_log_lik_over_lambdas(C, M, prop_over_lambdas, lambda_0, d, nSpots) + tf.transpose(tf_log_lik_over_lambdas_single_cell(C_sc, TC,  prop_over_lambdas, lambda_0, d, rho, rho_0)))
        
        r = tf.reduce_sum(log_prop_lik, axis=1) - tf.reduce_sum(log_curr_lik, axis=1) + tf.reduce_sum(bias_curr, axis=1) - tf.reduce_sum(bias_prop, axis=1)
    
        los = tf.random.uniform(shape=(nGenes,), dtype=tf.float32)
        decision = r > tf.math.log(los)
        curr_over_lambdas = tf.where(decision[:, tf.newaxis], prop_over_lambdas, curr_over_lambdas)
        
        return (curr_over_lambdas, decision, r, vector_curr_over_lambdas)
    
    def tf_proposal_lambda_0(curr_lambda_0, step_size_lambda_0):
        return tf_my_trunc_norm_sampling_lambda(curr_lambda_0, step_size_lambda_0)
    
    def tf_log_lik_lambda_0_single_cell(C_sc, TC, over_lambda, lambda_0, p_g, rho, rho_0, nCells):
        over_lambda = tf.transpose(tf.transpose(over_lambda) * rho)
        Lambda = over_lambda + lambda_0 * rho_0
        pg_factor = (1 - p_g) / p_g
        p_g_m = tf.tile(tf.expand_dims(p_g, 0), [nCells,1])
        r_gs = tf.transpose(tf.transpose(Lambda) * pg_factor)
        r_gs = tf.gather(r_gs, TC-1, axis=1)
        dist = tfp.distributions.NegativeBinomial(total_count=r_gs, probs=tf.transpose(p_g_m))
        A = tf.concat([ tf.cast(tf.reshape(TC, (1, -1)), dtype="float32"), dist.log_prob(C_sc)], axis=0)
        sorted_indices = tf.argsort(A[0, :])
        A = tf.gather(A, sorted_indices, axis=1)
        unique_elements, _ = tf.unique(A[0, :])
        BB = []
    
        for unique_element in unique_elements:
            mask = tf.equal(A[0, :], unique_element)
            split = tf.boolean_mask(A, mask, axis=1)
            BB.append(split)
            
        sums = []
        for tensor in BB:
            sum_result = tf.reduce_sum(tensor, axis=1, keepdims=True)
            sums.append(sum_result)
    
        temp = tf.concat(sums, axis=1)
        temp = tf.slice(temp, [1, 0], [-1, -1])
        
        return tf.reduce_sum(temp)
    
    def tf_log_lik_lambda_0(C, M, lambda_0, over_lambda, p, C_sc, TC, rho, rho_0, nSpots, nCells):
        Lambda = over_lambda + lambda_0
        mu_gs = tf.matmul(Lambda, tf.transpose(M))
        dist = tfp.distributions.Poisson(rate=mu_gs)
        return  tf.reduce_sum(dist.log_prob(C))   + tf_log_lik_lambda_0_single_cell(C_sc, TC, over_lambda, lambda_0, p, rho, rho_0, nCells)
    
    
    def tf_update_lambda_0(C, M, curr_lambda_0, over_lambda, d, step_size_lambda_0, C_sc, TC, rho, rho_0, nSpots, nCells):
        prop_lambda_0 = tf_proposal_lambda_0(curr_lambda_0, step_size_lambda_0)
        log_prop_lik = tf_log_lik_lambda_0(C, M, prop_lambda_0, over_lambda, d, C_sc, TC, rho, rho_0, nSpots, nCells)
        log_curr_lik = tf_log_lik_lambda_0(C, M, curr_lambda_0, over_lambda, d, C_sc, TC, rho, rho_0, nSpots, nCells)
        bias_prop = tf.cast(tfp.distributions.Normal(0, step_size_lambda_0).log_cdf(prop_lambda_0), tf.float32)
        bias_curr = tf.cast(tfp.distributions.Normal(0, step_size_lambda_0).log_cdf(curr_lambda_0), tf.float32)
        r = log_prop_lik - log_curr_lik  + bias_curr - bias_prop 
        los = tf.random.uniform(shape=[1])
        decision = r > tf.math.log(los)
        if (decision):
            curr_lambda_0 = prop_lambda_0
        return (curr_lambda_0, decision, r) 
    
    def random_choice_prob_index(a, axis=1):
        r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
        return (a.cumsum(axis=axis) > r).argmax(axis=axis)
    
    def tf_log_lik_TC_single_type(C_sc, TC, over_lambdas, lambda_0, p_g, rho, rho_0, nCells):
        rho = tf.reshape(rho, [nGenes])
        Lambda = tf.transpose(tf.transpose(over_lambdas) * rho) + lambda_0 * rho_0
        p_g_m = tf.tile(tf.expand_dims(p_g,0), (nCells,1))
        pg_factor = (1 - p_g) / p_g
        r_gs = tf.transpose(tf.transpose(Lambda) * pg_factor)
        r_gs = tf.transpose(tf.tile(tf.expand_dims(r_gs[:, TC - 1], axis=0), [nCells,1]))
        dist = tfp.distributions.NegativeBinomial(total_count=r_gs, probs=tf.transpose(p_g_m))
        log_likelihood = tf.reduce_sum(dist.log_prob(C_sc), axis=0)
        return log_likelihood.numpy()
    
    def tf_update_TC(C_sc, over_lambdas, lambda_0, p_g, rho, rho_0, D, nCells, nTypes):
        final = []
        for ii in range(0,nTypes):
            final.append(tf_log_lik_TC_single_type(C_sc, ii+1, over_lambdas, lambda_0, p_g, rho, rho_0, nCells))
            
        A = tf.convert_to_tensor(final, dtype=tf.float32)
        A = tf.transpose(A) + tf.math.log(D)
        uniform_samples = -tf.math.log(-tf.math.log(tf.random.uniform((nCells, nTypes))))
        v = uniform_samples + A
        res = tf.argmax(v, axis=1) + 1
        return res
    
    
    def tf_update_D(TC, d, nTypes):
        TC = tf.concat([TC, tf.range(1, nTypes + 1, dtype=tf.int64)], axis=0)
        unique,  idx, counts = tf.unique_with_counts(TC)
        counts = tf.cast(counts, "float32")
        d_prim = d + counts - 1
        return tf.convert_to_tensor(random.dirichlet(d_prim.numpy()), dtype="float32")
    
    
    def tf_proposal_rho(rho, step_size):
        return(tf_my_trunc_norm_sampling_vector(rho, step_size))
    
    def tf_update_rho(over_lambdas, lambda_0, p_g, step_size_rho, C_sc, TC, curr_rho, curr_rho_0):
        prop_rho = tf_proposal_rho(curr_rho, step_size_rho)
        
        log_curr_lik = tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, curr_rho_0, nCells), axis=1) + tfp.distributions.Normal(loc=rho, scale=0.1).log_prob(curr_rho)
        log_prop_lik = tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, prop_rho, curr_rho_0, nCells), axis=1) + tfp.distributions.Normal(loc=rho, scale=0.1).log_prob(prop_rho)
        
        log_bias_prop = tf.cast(tfp.distributions.Normal(0, step_size_rho).log_cdf(prop_rho), tf.float32)
        log_bias_curr = tf.cast(tfp.distributions.Normal(0, step_size_rho).log_cdf(curr_rho), tf.float32)
    
        r = log_prop_lik - log_curr_lik + log_bias_curr - log_bias_prop
        los = tf.random.uniform(shape=(nGenes,), dtype=tf.float32)
        decision = r > tf.math.log(los)
        
        prop_rho = tf.where(decision, prop_rho, curr_rho)
        return (prop_rho, decision, r)
    
    
    
    
    def tf_update_rho_0(over_lambdas, lambda_0, p_g, step_size_rho_0, C_sc, TC, curr_rho, curr_rho_0, prior_rho_0):
        prop_rho_0 = tf_proposal_lambda_0(curr_rho_0, step_size_rho_0)
        log_curr_lik = tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, curr_rho_0, nCells)) + tfp.distributions.Normal(loc=prior_rho_0, scale=0.1).log_prob(curr_rho_0)
        log_prop_lik =  tf.reduce_sum(tf_log_lik_p_g_single_cell(C_sc, TC, over_lambdas, lambda_0, p_g, curr_rho, prop_rho_0,nCells)) + tfp.distributions.Normal(loc=prior_rho_0, scale=0.1).log_prob(prop_rho_0)
        
        log_bias_prop = tf.reduce_sum(tfp.distributions.Normal(0, step_size_rho_0).log_cdf(prop_rho_0))
        log_bias_curr = tf.reduce_sum(tfp.distributions.Normal(0, step_size_rho_0).log_cdf(curr_rho_0))  
                
        r = log_prop_lik - log_curr_lik + log_bias_curr - log_bias_prop
        
        los = tf.random.uniform(shape=[1])
    
        decision = r > tf.math.log(los)
        if (decision):
            curr_rho_0 = prop_rho_0
    
        return (curr_rho_0, decision, r)
    
    
    def Gibbs():
    

        licznik = 0
        sum_M = np.zeros((nSpots,nTypes))
        
        current_lambda_0 = tf.convert_to_tensor(uniform.rvs(loc=0, scale=2, size=1), "float32")
        current_Z = tf.convert_to_tensor(np.ones((nSpots,nTypes)).astype(int), dtype="float32")
        current_thetas =  tf.convert_to_tensor(gamma.rvs(a=a*np.ones((nSpots,nTypes)), scale=b), dtype="float32")
    
        d = (1/nTypes)*np.ones((nTypes))
        current_D = tf.cast((1/nTypes)*np.ones(nTypes), dtype="float32")   
                
        current_pg =  tf.convert_to_tensor(uniform.rvs(loc=0, scale=1, size=nGenes, random_state=None), dtype="float32")
        current_rho = rho
        current_rho_0 = 2.5
        
        prob = np.tile(np.repeat(1/nTypes, nTypes), (nCells, 1))
        current_TC =  random_choice_prob_index(prob)+1

        current_N =  tf.convert_to_tensor(poisson.rvs(size=nSpots, mu=20), dtype="float32") 
        current_M = np.zeros((nSpots,nTypes))
        vec = np.repeat(1/nTypes,nTypes)
        for i in range(0,nSpots):
             current_M[i,:] = multinomial.rvs(current_N[i], vec, size=1)
        vec = np.repeat(1/nTypes,nTypes)
        func = lambda t:  multinomial.rvs(t, vec, size=1 )
        current_M = np.array(list(map(func, current_N)))[:, 0, :]
        current_M = tf.convert_to_tensor(current_M, dtype="float32")
            
        tf_step_size_N = tf.convert_to_tensor(st_N*np.ones(nSpots), dtype="float32")

        tf_step_size_M = tf.convert_to_tensor(st_M*np.ones( nSpots )  , dtype="float32")

        tf_step_size_thetas = tf.convert_to_tensor( st_thetas*np.ones((nSpots,nTypes)) ,    dtype="float32")

        step_size_pg = tf.convert_to_tensor(st_pg*np.ones(nGenes), dtype="float32")
        step_size_rho_0 = st_rho_0

        
        step_size_over_lambdas = prior_lambdas/5
        out = np.where(step_size_over_lambdas<2)
        step_size_over_lambdas[out] = 2
        step_size_over_lambdas = tf.convert_to_tensor(step_size_over_lambdas, dtype="float32")
        

        step_size_lambda_0 = tf.cast(st_lambda0, dtype="float32")
        step_sized_rho = tf.convert_to_tensor(st_rho*np.ones(nGenes), dtype="float32")
    
        indices_ones = tf.where(tf.equal(B, 1))
        indices_zeros = tf.where(tf.equal(B, 0))
        
        xx1 = indices_ones[:, 0]
        yy1 = indices_ones[:, 1]
        
        xx0 = indices_zeros[:, 0]
        yy0 = indices_zeros[:, 1]

        how_many_each_type = tf.reduce_sum(B, axis=1, keepdims=False)
        ind = np.array(B, dtype=bool)
    
        current_over_lambdas = np.zeros((nGenes, nTypes), dtype=float)
        current_over_lambdas[xx1,yy1] = lambdas_start/10
        current_over_lambdas = tf.convert_to_tensor(current_over_lambdas, dtype="float32")
        prev_S = tf.zeros([nSpots], tf.float32)
        zeros_column = tf.zeros([tf.shape(current_Z)[0], 1], dtype=current_Z.dtype)

        for i in range(0, number_of_it):
            print(i)
            
            beta_distribution = tfp.distributions.Beta(alpha / nTypes + current_Z, 2 - current_Z)
            current_pi = beta_distribution.sample()
            current_Z = tf_update_Z(current_thetas, current_pi, a, b, a_0, b_0)
            
            current_Z = tf.concat([current_Z[:, :-1], zeros_column], axis=1)
            current_thetas = tf_update_thetas(current_M, current_thetas, current_Z, current_over_lambdas + current_lambda_0, current_N, tf_step_size_thetas, a, b, a_0, b_0, nTypes, nSpots)
            row_sums = tf.reduce_sum(current_thetas, axis=1)
            expanded_row_sums = tf.expand_dims(row_sums, axis=1)
            current_H = current_thetas / expanded_row_sums
            res_N_M = tf_update_N_M(current_N, current_M, tf_C_gs, current_H, current_over_lambdas + current_lambda_0, current_pg, tf_step_size_N,  tf_step_size_M, prev_S, n_cells)
            current_M = res_N_M[1]
            current_N = res_N_M[0]
            prev_S =  res_N_M[4]
            
            res_over_lambdas = tf_update_over_lambdas(tf_C_gs, tf_B, current_M, current_over_lambdas, current_lambda_0, current_pg, step_size_over_lambdas, xx0, yy0, xx1, yy1, ind, how_many_each_type, tf_C_gc, current_TC, current_rho, current_rho_0, indices_ones, indices_zeros)
            current_over_lambdas = res_over_lambdas[0]
           
            res_lambda_0 = tf_update_lambda_0(tf_C_gs, current_M, current_lambda_0, current_over_lambdas, current_pg, step_size_lambda_0, tf_C_gc, current_TC, current_rho, current_rho_0, nSpots, nCells)
            current_lambda_0 = res_lambda_0[0] 

            res_pg = tf_update_p_g(tf_C_gs, current_over_lambdas, current_lambda_0, current_pg,  current_M, step_size_pg, tf_C_gc, current_TC, current_rho, current_rho_0, nSpots, nCells)
            current_pg = res_pg[0]
    
            current_TC = tf_update_TC(tf_C_gc, current_over_lambdas, current_lambda_0, current_pg, current_rho, current_rho_0, current_D, nCells, nTypes)
            current_D = tf_update_D(current_TC, d, nTypes)
            
            res_rho = tf_update_rho(current_over_lambdas, current_lambda_0, current_pg, step_sized_rho, tf_C_gc, current_TC, current_rho, current_rho_0)
            current_rho = res_rho[0]
            
            
            res_rho_0 = tf_update_rho_0(current_over_lambdas, current_lambda_0, current_pg, step_size_rho_0, tf_C_gc, current_TC, current_rho, current_rho_0, prior_rho_0)
            current_rho_0 = res_rho_0[0]
        
            
            if i%100==0:
                pd.DataFrame(current_M.numpy().reshape(nSpots*nTypes)).T.to_csv( address_results + "/res_M.csv", header=False, index=False, mode="a")
                pd.DataFrame(current_TC).T.to_csv(address_results + "/res_TC.csv", header=False, index=False, mode="a")

            if i>burn_in:
                licznik = licznik + 1
                sum_M = sum_M + current_M
                   
            
        pd.DataFrame(sum_M/licznik).to_csv(address_results + "/est_M.csv", header=False, index=False)
            

    Gibbs()
