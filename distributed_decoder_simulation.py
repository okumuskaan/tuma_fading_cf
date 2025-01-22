"""
Distributed Decoder Simulation

This script implements and simulates the distributed decoder for the TUMA framework with a fading 
cell-free massive MIMO setup. It uses AMP iterations with Bayesian estimation.

Author:
Kaan Okumus
Date:
January 2025
"""

import numpy as np
from utils.helper_bayesian_denoiser import bayesian_channel_multiplicity_estimation_sampbasedapprox
from utils.helper_topology import setup_topology
from utils.helper_cf_tuma_tx import transmit
from centralized_decoder_simulation import generate_priors, generate_all_covs

def obtain_real_type_from_k(k, U):
    real_k = k.reshape(U,-1).sum(axis=0)
    real_type = real_k/(real_k.sum())
    return real_type

class AP:
    def __init__(self, id, A, Cx, Cy, U, Mus, n, P, nAMPiter, Y, all_covs, priors, log_priors, X_true):
        self.id = id
        self.A = A
        self.Cx = Cx
        self.Cy = Cy
        self.U = U
        self.Mus = Mus
        self.n = n
        self.P = P
        self.nP = n*P
        self.nAMPiter = nAMPiter
        self.all_covs = all_covs
        self.priors = priors
        self.log_priors = log_priors
        self.Y = Y
        self.X_true = X_true

    def __str__(self):
        return f"AP {self.id+1}"

    def AMP_decoder(self):
        self.Z = self.Y.copy()
        self.est_X_0 = [np.zeros((self.Mus[u],self.A), dtype=complex) for u in range(self.U)]
        self.est_X = [np.zeros((self.Mus[u],self.A), dtype=complex) for u in range(self.U)]

        self.tv_dists = []

        print(f"\t\tAP{self.id+1} AMP decoder ...")

        for t in range(self.nAMPiter):
            print(f"\t\t\tAMPiter = {t+1}/{self.nAMPiter}")
            # Covariance matrix of residual noise
            taus = np.mean(np.diag(self.Z.conj().T @ self.Z).real/self.n)
            self.T = np.ones(self.A)*taus

            self.Gamma = np.zeros_like(self.Z)
            self.R = []
            for u in range(self.U):
                #print(f"\t\tu = {u+1}/{self.U}")
                
                # Compute effective observation for zone u
                R_u = self.Cy(self.Z, u) + np.sqrt(self.nP) * self.est_X[u]
                self.R.append(R_u)
                
                # Compute denoiser and Q (for Onsager reaction term) for zone u
                Qu = np.zeros((self.A, self.A), dtype=complex)
                for m in range(self.Mus[u]):
                    self.est_X[u][m] = bayesian_channel_multiplicity_estimation_sampbasedapprox(R_u[m], self.all_covs[u], self.T, self.nP, self.log_priors[u][m])
                    
                # Compute residual 
                self.Gamma += self.Cx(self.est_X[u], u)
                Qu /= self.Mus[u]
                self.Gamma -= (self.Mus[u] / self.n) * self.Z @ Qu
            self.Z = self.Y - np.sqrt(self.nP) * self.Gamma

        self.log_likelihoods = self.compute_local_likelihood(self.R, self.T, self.all_covs, self.Mus, self.nP)

    
    def normalize_posteriors(self, logposteriors):
        max_posterior = np.max(logposteriors)
        log_sumposteriors = max_posterior + np.log(np.exp(logposteriors - max_posterior).sum())
        return np.exp(logposteriors - log_sumposteriors)
    
    def compute_loglikelihoods_with_logsumexptrick(self, y, all_Covs, T, nP, Ns):
        logai = (-((np.abs(y)**2)/(T + nP * all_Covs) + np.log(np.pi * (T + nP * all_Covs))).sum(axis=-1) - np.log(Ns))
        maxlogai = np.max(logai,axis=-1, keepdims=True)
        loglikelihoods = (maxlogai[:,0] + np.log(np.exp(logai - maxlogai).sum(axis=-1)))
        return loglikelihoods

    def compute_local_likelihood(self, R, T, all_Covs, Mus, nP):
        loglikelihoods = []
        Kmax = all_Covs.shape[1]
        Ns = all_Covs.shape[-2]
        for u, Mu in enumerate(Mus):
            loglikelihoods_u = np.zeros((Mu, Kmax))
            for mu in range(Mu):
                loglikelihoods_um = self.compute_loglikelihoods_with_logsumexptrick(R[u][mu], all_Covs[u], T, nP, Ns)
                loglikelihoods_u[mu] = loglikelihoods_um
            loglikelihoods.append(loglikelihoods_u)
        return loglikelihoods



class CPU:
    def __init__(self, Y, U, A, B, all_Covs, priors, log_priors, Cx, Cy, Mus, n, P, nAMPiter, Xs_true):
        self.A = A
        self.B = B
        self.F = A*B
        self.Y = Y
        self.U = U
        self.log_priors = log_priors
        self.APs = []
        self.Mu = Mus[0]

        for b in range(B):
            self.APs.append(
                AP(id=b, A=A, Cx=Cx, Cy=Cy, U=U, Mus=Mus, n=n, P=P, 
                   nAMPiter=nAMPiter, Y=Y[:,b*A:(b+1)*A], all_covs=all_Covs[:,:,:,b*A:(b+1)*A], 
                   priors=priors,
                   log_priors=log_priors, X_true=Xs_true[b])
            )

    def AMP_decoder(self):
        for b in range(self.B):
            self.APs[b].AMP_decoder()

    def compute_local_loglikelihoods(self):
        self.log_likelihoods_across_users = []
        for b in range(self.B):
            self.log_likelihoods_across_users.append(self.APs[b].log_likelihoods)
        self.log_likelihoods_across_users = np.array(self.log_likelihoods_across_users)

    def aggregate_and_estimate_types(self):
        self.log_posteriors = np.sum(self.log_likelihoods_across_users, axis=0) + self.log_priors
        self.est_k = np.zeros(self.U*self.Mu)
        for u in range(self.U):
            for m in range(self.Mu):
                self.est_k[u*self.Mu + m] = np.argmax(self.log_posteriors[u,m])



def simulate_distributed_decoder(
    side, SNR_rx_dB, Ju, Mau, Kau, nAMPIter, A, n, P, nMCs, d0, rho,
                        force_Kmax=3, Kmax=5, Ns=500, display_topology=False):

    tv_dists = np.zeros(nMCs)
    tv_dists_new = np.zeros(nMCs)

    for idxMC in range(nMCs):
        display_topology = False if idxMC>0 else display_topology
        print(f"\tMonte Carlo Sim = {idxMC+1}/{nMCs}")

        U, B, zone_centers, nus = setup_topology(
            side=side, 
            topology_type=2, 
            mult=2, jitter=0.0, multiple_zone=True,
            rows=3, cols=3
        )
        F = A*B

        Maus = [Mau]*U
        Kaus = [Kau]*U
        Mu = 2**Ju
        Mus = [Mu]*U

        SNR_rx = 10**(SNR_rx_dB/10)
        #print(f"SNR_rx_dB = {SNR_rx_dB}, Mu = {Mu}")
        min_dist = np.abs(np.array([(1+1j)*0.0]) - nus).min()
        SNR_tx = SNR_rx * (1 + (min_dist/d0)**rho)
        sigma_w = np.sqrt(P/SNR_tx)
        nP = n * P

        # Prepare the codebook
        M = sum(Mus)
        C = np.random.randn(n,M) + 1j * np.random.randn(n,M)
        C = C/np.linalg.norm(C,axis=0)
        def Cx(x,u=0):
            return C[:,u*Mus[u]:(u+1)*Mus[u]]@x
        def Cy(y,u=0):
            return (C[:,u*Mus[u]:(u+1)*Mus[u]].conj().T)@y
        
        # Transmitter
        topology_type = 2
        margin = side/10 if topology_type!=3 else -side/20
        Y, k, X, _ = transmit(Mus, Maus, Kaus, side, n, F, A, U, Cx, nP, sigma_w, 
                                nus, zone_centers, rho, d0, margin=margin, force_Kmax=force_Kmax)
        
        Xs = []
        for b in range(B):
            sub_X = []
            for u in range(U):
                sub_X.append(X[u].reshape(-1, B, A)[:,b,:])
            Xs.append(sub_X)

        # Compute prior
        priors = generate_priors(
            Kaus, Maus, Mus, Kmax=Kmax
        )
        log_priors = [np.log(priors_u) for priors_u in priors]

        # Compute covariances for sampling-based approximation
        _, all_Covs = generate_all_covs(side, nus, zone_centers, A, F, rho, d0, Kmax=Kmax, Ns=Ns)

        cpu = CPU(Y=Y, U=U, A=A, B=B, all_Covs=all_Covs, priors=priors, log_priors=log_priors, Cx=Cx, Cy=Cy, Mus=Mus, n=n, P=P, nAMPiter=nAMPIter, Xs_true=Xs)

        cpu.AMP_decoder()

        cpu.compute_local_loglikelihoods()

        cpu.aggregate_and_estimate_types()

        tv_dist = np.sum(np.abs(k/np.sum(k) - cpu.est_k/np.sum(cpu.est_k)))/2
        tv_dist_new = np.abs(obtain_real_type_from_k(k,U) - obtain_real_type_from_k(cpu.est_k,U)).sum()/2

        tv_dists[idxMC] = tv_dist
        tv_dists_new[idxMC] = tv_dist_new

        print(f"\ttv_dist = {tv_dist}, tv_dist_new = {tv_dist_new}")

    return tv_dists_new

