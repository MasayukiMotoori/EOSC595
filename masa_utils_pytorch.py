import numpy as np
from scipy import optimize
from scipy.constants import mu_0, epsilon_0
from scipy import fftpack
from scipy import sparse
from scipy.special import factorial
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, CubicSpline,splrep, BSpline
from scipy.sparse import csr_matrix, csc_matrix
import csv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import lu_factor, lu_solve
from scipy import signal
import empymod
import discretize
import  os
eps= np.finfo(float).eps
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod

class Pelton_res_f():
    def __init__(self, freq=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = freq
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def f(self,p):
        """
        Pelton resistivity model made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta
        p[2] : log(tau)
        p[3] : c
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        f = torch.exp(p[0])*(tc +(1.0-p[1])*iwc)/(tc+iwc)
        return f        

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = 2
        ind_c = 3
       
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton_res_f_two():
    def __init__(self, freq=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = freq
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)
    
    def f(self,p):
        """
        Pelton two relaxation resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta
        p[2] : log(tau1)
        p[3] : c1
        p[4] : log(tau2)
        p[5] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[5]
        tc2 = torch.exp(-p[4]*p[5])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)*(tc2)/(tc1+iwc1)/(tc2+iwc2)
        return f
    
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1
        ind_tau = [2,4]
        ind_c = [3,5] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class Pelton_res_f_dual():
    def __init__(self, freq=None,
        reslim= [1e-2,1e5],
        chglim= [1e-3, 0.9],
        taulim= [1e-8, 1e4],
        clim= [0.2, 0.8]
            ):
        self.freq = freq
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def pelton_res_f_dual(self,p):
        """
        Pelton dual cole-cole resistivity model
        made easy for PyTorch Auto Diffentiation.
        p[0] : log(res0)
        p[1] : eta1
        p[2] : log(tau1)
        p[3] : c1
        p[4] : eta2
        p[5] : log(tau2)
        p[6] : c2
        """
        iwc1 = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc1 = torch.exp(-p[2]*p[3])
        iwc2 = (1j * 2. * torch.pi * self.freq  ) ** p[6]
        tc2 = torch.exp(-p[5]*p[6])
        f = torch.exp(p[0])*(tc1 +(1.0-p[1])*iwc1)/(tc1+iwc1)*(tc2 +(1.0-p[4])*iwc2)/(tc2+iwc2)
        return f

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = [1,4]
        ind_tau = [2,5]
        ind_c = [3,6] 
    
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp
    
class Pelton_con_f():
    def __init__(self, freq=None,
        conlim= [1e-2,1e5],
        chglim= [1e-3, 0.9],
        taulim= [1e-8, 1e4],
        clim= [0.2, 0.8]
            ):
        self.freq = freq
        self.conlim = torch.tensor(np.log(conlim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def f(self,p):
        """
        Pelton conductivity model made easy for PyTorch Auto Diffentiation.
        p[0] : log(con8)
        p[1] : eta
        p[2] : log(tau)
        p[3] : c
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        f = torch.exp(p[0])*(1.0-p[1])*(tc +iwc)/(tc+(1.0-p[1])*iwc)
        return f
    
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_con = 0
        ind_chg = 1
        ind_tau = 2
        ind_c = 3 
    
        mvec_tmp[ind_con] = torch.clamp(mvec[ind_con], self.conlim.min(), self.conlim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
        mvec_tmp[ind_c] = torch.clamp(mvec[ind_c], self.clim.min(), self.clim.max())
        return mvec_tmp

class BaseSimulation:
    eps = torch.finfo(torch.float32).eps
    @abstractmethod
    def dpred(self,m):
        pass
    @abstractmethod
    def J(self,m):
        pass
    @abstractmethod
    def project_convex_set(self,m):
        pass

class InducedPolarizationSimulation(BaseSimulation):
    eps = torch.finfo(torch.float32).eps
    def __init__(self, 
                 t=False,
                 times=None,
                 ip_model=None,
                 window_mat=None,
                 windows_strt=None,
                 windows_end=None
                 ):
        self.t = t
        self.times = times
        self.ip_model = ip_model
        self.window_mat = window_mat 
        self.windows_strt = windows_strt
        self.windows_end = windows_end 
   
    def count_data_windows(self, times):
        nwindows = len(self.windows_strt)
        count_data = torch.zeros(nwindows)
        for i in torch.arange(nwindows):
            start = self.windows_strt[i]
            end = self.windows_end[i]
            ind_time = (times >= start) & (times <= end)
            count_data[i] = ind_time.sum()
        return count_data

    def get_freq_windowmat(self,tau,log2min=-8, log2max=8):
        # freqend = (1/tau)*2**log2max
        # freqstep = (1/tau)*2**log2min
        # freq = torch.arange(0,freqend,freqstep)
        freqend = ((1 / tau) * 2 ** log2max).item()
        freqstep = ((1 / tau) * 2 ** log2min).item()
        freq = torch.arange(0, freqend, freqstep)
        times = torch.arange(0, 1/freqstep,1/freqend)
        count = self.count_data_windows(times)
        while count.min() < 2:
            print(count)
            log2max += 1
            freqend = ((1/tau)*2**log2max).item()
            if log2max > 15:
                print('Some windows has less than two data')
                break
            if count[self.windows_end==self.windows_end.max()] <2:
                log2min -= 1
                freqstep = ((1/tau)*2**log2min).item()
                if log2min < -15:
                    print('Some windows has less than two data')
                    break
            freq = torch.arange(0,freqend,freqstep)
            times = torch.arange(0, 1/freqstep,1/freqend)
            count = self.count_data_windows(times)
        self.times = times
        self.ip_model.freq = freq
        self.get_window_matrix(times=times)
        
    def freq_symmetric(self,f):
        f_sym = torch.zeros_like(f, dtype=torch.cfloat)
        nfreq = len(f)
        if nfreq  %2 == 0:
            nfreq2 = nfreq // 2
            f_sym[:nfreq2] = f[:nfreq2]
            f_sym[nfreq2] =f[nfreq2].real
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2].conj(), dims=[0])

        # Ensure symmetry at the Nyquist frequency (if even length)
        else:
            nfreq2 = nfreq // 2 
            f_sym[:nfreq2+1] = f[:nfreq2+1]
            f_sym[nfreq2+1:] = torch.flip(f[1:nfreq2+1].conj(), dims=[0])
        return f_sym

    def compute_fft(self,f):
        f_sym = self.freq_symmetric(f)
        t = torch.fft.ifft(f_sym) 
        return t

    def get_windows(self,windows_cen):
        windows_cen = windows_cen[windows_cen>0]
        windows_strt = torch.zeros_like(windows_cen)
        windows_end  = torch.zeros_like(windows_cen)
        dt = torch.diff(windows_cen)
        windows_strt[1:] = windows_cen[:-1] + dt / 2
        windows_end[:-1] = windows_cen[1:] - dt / 2
        windows_strt[0] = 10*eps
        windows_end[-1] = windows_cen[-1] + dt[-1] / 2
        self.windows_strt = windows_strt
        self.windows_end = windows_end

    def get_window_matrix (self,times, sum=False):
        nwindows = len(self.windows_strt)
        window_matrix = torch.zeros((nwindows+1, len(times)))
        window_matrix[0,0] = 1
        for i in range(nwindows):
            ind_time = (times >= self.windows_strt[i]) & (times <= self.windows_end[i])
            if ind_time.sum() > 0:
                if sum:
                    window_matrix[i+1, ind_time] = torch.ones(ind_time.sum())
                window_matrix[i+1, ind_time] = 1.0/(ind_time.sum())
        self.window_mat = window_matrix

    def dpred(self,m):
        if self.t:
            self.get_freq_windowmat(tau=torch.exp(m[2]))
            f = self.ip_model.f(m)
            t = self.compute_fft(f)/(self.times[1]-self.times[0])
            t_real = t.real
            return self.window_mat@t_real
        
        else:
            f = self.ip_model.f(m)
            f_real = f.real
            f_imag = f.imag
            if self.window_mat is None:
                return torch.cat([f_real, f_imag])
            else:
                return torch.cat([self.window_mat@f_real, self.window_mat@f_imag])

    def J(self,m):
        return torch.autograd.functional.jacobian(self.dpred, m)   

    def project_convex_set(self,m):
        return self.ip_model.clip_model(m)

class Optimization():  # Inherits from BaseSimulation
    def __init__(self,
                sim,
                dobs=None,
                Wd=None, Ws=None, Wx=None, alphas=1, alphax=1,
                Ws_threshold=1e-12
                ):
        self.sim = sim  # Composition: opt_tmp has a simulation
        self.dobs= dobs
        self.Wd = Wd
        self.Ws = Ws
        self.Ws_threshold = Ws_threshold
        self.Wx = Wx
        self.alphas = alphas
        self.alphax = alphax
    
    def dpred(self, m):
        return self.sim.dpred(m)  # Calls InducedPolarization's dpred()

    def J(self, m):
        return self.sim.J(m)  # Calls InducedPolarization's J()
    
    def project_convex_set(self,m):
        return self.sim.project_convex_set(m)

    def get_Wd(self,ratio=0.10, plateau=0):
        dobs_clone = self.dobs.clone().detach()
        noise_floor = plateau * torch.ones_like(dobs_clone)
        std = torch.sqrt(noise_floor**2 + (ratio * torch.abs(dobs_clone))**2)
        self.Wd =torch.diag(1 / std.flatten())
        return self.Wd
    
    def get_Ws(self, mvec):
        self.Ws = torch.eye(mvec.shape[0])
        return self.Ws
    
    def loss_func(self,m, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * torch.dot(r,r)
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = 0.5 * torch.dot(rms, rms)
            return phid + phim
        return phid
    
    def BetaEstimate_byEig(self,mvec, beta0_ratio=1.0, eig_tol=eps, update_Wsen=False):
        alphax=self.alphax
        alphas=self.alphas
        Wd = self.Wd
        Wx = self.Wx
        Ws= self.Ws
        J = self.J(mvec)

        if update_Wsen:
            self.update_Ws(J)    

        # Prj_m = self.Prj_m  # Use `Proj_m` to map the model space

        # Effective data misfit term with projection matrix
        A_data =  J.T @ Wd.T @ Wd @ J 
        eig_data = torch.linalg.eigvalsh(A_data)
        
        # Effective regularization term with projection matrix
        # A_reg = alphax* Prj_m.T @ Wx.T @ Wx @ Prj_m
        A_reg = torch.zeros_like(A_data)
        if Wx is not None:
            A_reg += alphax * Wx.T @ Wx 
        if Ws is not None:
            A_reg += alphas * (Ws.T @ Ws)
        eig_reg = torch.linalg.eigvalsh(A_reg)
        
        # Ensure numerical stability (avoid dividing by zero)
        eig_data = eig_data[eig_data > eig_tol]
        eig_reg = eig_reg[eig_reg > eig_tol]

        # Use the ratio of eigenvalues to set beta range
        lambda_d = torch.max(eig_data)
        lambda_r = torch.min(eig_reg)
        return beta0_ratio * lambda_d / lambda_r
  
    def compute_sensitivity(self,J):
        return  torch.sqrt(torch.sum(J**2, axis=0))

    def update_Ws(self, J):
        Wd=self.Wd
        Sensitivity = self.compute_sensitivity(Wd@J)
        Sensitivity /= Sensitivity.max()
        Sensitivity = np.clip(Sensitivity, self.Ws_threshold, 1)
        Ws = torch.diag(Sensitivity)
        self.Ws = Ws
        return Ws
    
    def GaussNewton(self,mvec_init, niter, beta0, print_update=True, coolingFactor=2.0, coolingRate=2,
        s0=torch.tensor(1.0), sfac = torch.tensor(0.5),update_Wsen=False, 
        stol=torch.tensor(1e-6), gtol=torch.tensor(1e-3), mu=torch.tensor(1e-4)):

        mvec_old = mvec_init.detach()
        m_ref = mvec_init
        f_old = self.loss_func(mvec_old, m_ref=m_ref)
        error_prg = torch.zeros(niter + 1)
        mvec_prg = torch.zeros((niter + 1, mvec_init.shape[0]))
        error_prg[0] = f_old
        mvec_prg[0, :] = mvec_old

        for i in range(niter):
            beta = beta0 / (coolingFactor ** (i // coolingRate))
            rd = self.Wd@(self.dpred(mvec_old) - self.dobs)
            J = self.J(mvec_old)
            if update_Wsen:
                self.update_Ws(J)                
            g = J.T @ self.Wd.T@ rd
            H = J.T @ self.Wd.T@ self.Wd@J
            if m_ref is not None:
                g += beta * self.alphas * (self.Ws.T@self.Ws@ (mvec_old - m_ref))
                H += beta * self.alphas * self.Ws.T@self.Ws
            if self.Wx is not None:
                g += beta * self.alphax * (
                    self.Wx.T @ self.Wx @ mvec_old)
                H += beta * self.alphax * self.Wx.T @ self.Wx
       
            
            dm = torch.linalg.solve(H, g).flatten()  # Ensure dm is a 1D tensor
            g_norm = torch.linalg.norm(g, ord=2)

            if g_norm < gtol:
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = s0
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            f_new = self.loss_func(mvec_new, m_ref=m_ref)# phid
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            while f_new >= f_old + s * mu * directional_derivative:
                s *= sfac
                mvec_new = self.project_convex_set(mvec_old - s * dm)
                f_new = self.loss_func(mvec_new,m_ref=m_ref) #phid
                if s < stol:
                    break
            mvec_old = mvec_new
            mvec_prg[i + 1, :] = mvec_new
            f_old = f_new
            error_prg[i + 1] = f_new
            if print_update:
                print(f'{i + 1:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e},  f:{f_new:.1e}')
        return mvec_new, error_prg, mvec_prg
    
class TEM_Signal_Process:
    
    def __init__(self,  
        base_freq,on_time, rmp_time, rec_time, smp_freq,
        windows_cen=None, windows_strt = None, windows_end = None):
        self.base_freq = base_freq
        self.on_time = on_time
        self.rmp_time = rmp_time
        self.rec_time = rec_time
        self.smp_freq = smp_freq
        time_step = 1./smp_freq
        self.time_step = time_step
        self.times_rec = np.arange(0,rec_time,time_step) + time_step
        self.times_filt = np.arange(0,rec_time,time_step)
        self.windows_cen= windows_cen
        self.windows_strt = windows_strt
        self.windows_end = windows_end
