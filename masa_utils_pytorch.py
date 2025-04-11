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
import  os
eps= np.finfo(float).eps
import torch
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from abc import ABC, abstractmethod

class TorchHelper:
    @staticmethod
    def to_tensor(x, dtype=torch.float32, device='cpu'):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype=dtype, device=device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)

class Pelton_res_f():
    def __init__(self, freq=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq = torch.tensor(freq, dtype=torch.cfloat)
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

class Pelton_debye_f():
    def __init__(self, freq=None,times=None,ntau=None,taus=None,
            reslim= [1e-2,1e5],
            chglim= [1e-3, 0.9],
            taulim= [1e-5, 1e1],
            taulimspc = [2,10],
                ):
        self.times = torch.tensor(times)
        self.freq = torch.tensor(freq, dtype=torch.cfloat)
        self.ntau = ntau
        self.taus = torch.tensor(taus) if taus is not None else None
        self.reslim = torch.tensor(np.log(reslim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.taulimspc = torch.tensor(np.log(taulimspc))

    def f(self, p):
        """
        Pelton linear weighted Debye model in frequency domain.
        Parameters:
            p[0]: log(rho0)
            p[1:1+ntau]: etas (relaxation weights)
            p[1+ntau:1+2*ntau]: log(taus) (if taus not fixed)
        Returns:
            Complex-valued resistivity at each frequency.
        """
        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau].to(dtype=torch.cfloat)
        if self.taus is None:
            assert len(p) == 1 + 2 * self.ntau
            taus = torch.exp(p[1 + self.ntau:1 + 2 * self.ntau])
        else:
            assert len(p) == 1 + self.ntau
            taus = self.taus.to(dtype=torch.cfloat)
        omega = 2 * torch.pi * self.freq
        omega = omega.view(-1, 1)  # shape: [nfreq, 1]
        taus = taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        iwt = 1j * omega * taus  # shape: [nfreq, ntau]
        term = -iwt * etas / (1 + iwt)  # shape: [nfreq, ntau]
        return rho0 * (1 + term.sum(dim=1))  # shape: [nfreq]

    def t(self, p):
        """
        Pelton linear weighted Debye model in time domain.
        Parameters:
            p[0]: log(rho0)
            p[1:1+ntau]: etas (relaxation weights)
            p[1+ntau:1+2*ntau]: log(taus) (if taus not fixed)
        Returns:
            Real value resistivity given times.
        """
        rho0 = torch.exp(p[0])
        etas = p[1:1 + self.ntau]
        if self.taus is None:
            assert len(p) == 1 + 2 * self.ntau
            taus = torch.exp(p[1 + self.ntau:1 + 2 * self.ntau])
        else:
            assert len(p) == 1 + self.ntau
            taus = self.taus.to(dtype=torch.float)
        
        ind_0 = torch.where(self.times == 0)[0]
        times = self.times.view(-1, 1)  # shape: [ntime, 1]
        taus = taus.view(1, -1)  # shape: [1, ntau]
        etas = etas.view(1, -1)  # shape: [1, ntau]
        term = etas/taus*torch.exp(-times/taus)  # shape: [ntime, ntau]
        term_sum = term.sum(dim=1)  # shape: [ntime]
        term_sum[ind_0] = 1-etas.sum(dim=1)  # Set the value at t=0 to 1 - sum(etas)

        return rho0 * (term_sum)  # shape: [tau]

    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()
        ind_res = 0
        ind_chg = 1+np.arange(self.ntau)
     
        mvec_tmp[ind_res] = torch.clamp(mvec[ind_res], self.reslim.min(), self.reslim.max())
        mvec_tmp[ind_chg] = torch.clamp(mvec[ind_chg], self.chglim.min(), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg], torch.ones(self.ntau), self.chglim.max())
        mvec_tmp[ind_chg] = self.proj_halfspace(mvec_tmp[ind_chg],-torch.ones(self.ntau), self.chglim.min())
        if self.taus is None:
            ind_tau = 1+self.ntau+np.arange(self.ntau)
            mvec_tmp[ind_tau] = torch.clamp(mvec[ind_tau], self.taulim.min(), self.taulim.max())
            mvec_tmp[ind_tau[0]] = torch.clamp(mvec[ind_tau[0]],
                        self.taulim.min()+self.taulimspc.min(), self.taulim.min() + self.taulimspc.max())
            a_local = torch.tensor(np.r_[-1.0,1.0], dtype=torch.float32)
            for i in range(self.ntau-1):
                mvec_tmp[ind_tau[i:i+2]] = self.proj_halfspace(
                    mvec[i:i+2], -a_local, self.taulimspc.max())
                mvec_tmp[ind_tau[i:i+2]] = self.proj_halfspace(
                    mvec[i:i+2],  a_local, self.taulimspc.min())
            mvec_tmp[ind_tau[-1]] = torch.clamp(mvec[ind_tau[-1]],
                        self.taulim.max()-self.taulimspc.max(), self.taulim.min() - self.taulimspc.min())
        return mvec_tmp

    def proj_halfspace(self, x, a, b):
        ax = torch.dot(a, x)
        if ax > b:
            proj_x = x + a * ((b - ax) / torch.dot(a, a))
        else:
            proj_x = x
        return proj_x


class ColeCole_f():
    def __init__(self, freq=None, res=False,
            conlim= [1e-5,1e2],
            chglim= [1e-3, 0.9],
            taulim= [1e-8, 1e4],
            clim= [0.2, 0.8]
                ):
        self.freq =  torch.tensor(freq, dtype=torch.cfloat)
        self.res = res
        self.reslim = torch.tensor(np.log(conlim))
        self.chglim = torch.tensor(chglim)
        self.taulim = torch.tensor(np.log(taulim))
        self.clim = torch.tensor(clim)

    def f(self,p):
        """
        Cole Cole conductivity model made easy for PyTorch Auto Diffentiation.
        p[0] : log(con8)
        p[1] : eta
        p[2] : log(tau)
        p[3] : c
        """
        iwc = (1j * 2. * torch.pi * self.freq  ) ** p[3] 
        tc = torch.exp(-p[2]*p[3])
        if self.res:
            return torch.exp(-p[0])/((1.0-p[1])*tc +iwc)*(tc+iwc)
        else:
            return torch.exp( p[0])*((1.0-p[1])*tc +iwc)/(tc+iwc)
        
    def clip_model(self,mvec):
        # Clone to avoid modifying the original tensor
        mvec_tmp = mvec.clone().detach()       
        mvec_tmp[0] = torch.clamp(mvec[0], self.reslim.min(), self.reslim.max())
        mvec_tmp[1] = torch.clamp(mvec[1], self.chglim.min(), self.chglim.max())
        mvec_tmp[2] = torch.clamp(mvec[2], self.taulim.min(), self.taulim.max())
        mvec_tmp[3] = torch.clamp(mvec[3], self.clim.min(), self.clim.max())
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

    def f(self,p):
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
        self.freq =  torch.tensor(freq, dtype=torch.cfloat)
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
                 ip_model=None,
                 t=False,
                 mode=None,
                 times=None,
                 basefreq=None,
                #  window_mat=None,
                #  windows_strt=None,
                #  windows_end=None
                 ):
        self.ip_model = ip_model
        self.t = t
        self.times = torch.tensror(times)
        self.basefreq = torch.tensror(basefreq)
        # self.window_mat = window_mat 
        # self.windows_strt = windows_strt
        # self.windows_end = windows_end 
   
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
        if self.mode=="tdip_t" :
            ip_t=self.ip_model.t(m)
            volt = torch.conv1d(ip_t,self.curr)
            return self.window_mat@volt
        
        elif self.mode=="tdip_f":
            self.get_freq_windowmat(tau=torch.exp(m[2]))
            ip_f=self.ip_model.f(m)
            ip_t = self.compute_fft(ip_f)
            ip_t = ip_t.real
            volt = torch.conv1d(ip_t,self.curr)
            return self.window_mat@volt

        elif self.mode=="sip_t":
            self.get_freq_windowmat(tau=torch.exp(m[2]))
            f = self.ip_model.f(m)
            t = self.compute_fft(f)/(self.times[1]-self.times[0])
            t_real = t.real
            return self.window_mat@t_real
        
        elif self.mode=="sip":
            f = self.ip_model.f(m)
            f_real = f.real
            f_imag = f.imag
            if self.window_mat is None:
                return torch.cat([f_real, f_imag])
            else:
                return torch.cat([self.window_mat@f_real, self.window_mat@f_imag])
        
        print("Simulation mode not available")
        return None

    def set_current_wave(self, basefreq):
        self.basefreq = basefreq
        self.curr=torch.tensor(0.5*(1.0+signal.square(2*np.pi*(self.basefreq*self.times),duty=0.5)))

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
    
    def loss_func_L2(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * torch.dot(r,r)
        phim = 0

        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = 0.5 * self.alphas*torch.dot(rms, rms)
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += 0.5 * self.alphax*torch.dot(rmx, rmx) 
        return phid+beta*phim, phid, phim 

    def loss_func_L1reg(self,m, beta, m_ref=None):
        r = self.dpred(m)-self.dobs
        r = self.Wd @ r
        phid = 0.5 * torch.dot(r,r)
        phim = 0
        if m_ref is not None:
            rms = self.Ws @ (m - m_ref)
            phim = self.alphas*torch.sum(abs(rms))
        if self.Wx is not None:
            rmx = self.Wx @ m
            phim += self.alphax*torch.sum(abs(rmx)) 
        return phid+beta*phim, phid, phim 

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

    def GradientDescentL1reg(self,mvec_init, niter, beta0, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4,ELS=True, BLS=True ):

        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        mvec_old = mvec_init
        m_ref = mvec_init.detach()
        beta= beta0


        for i in range(niter):
            beta =  beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))
            J = self.J(mvec_old)
            if update_Wsen:
                self.update_Ws(J) 

            f_old, phid, phim = self.loss_func_L1reg(mvec_old,beta=beta,m_ref=m_ref) 
            f_old.backward()   # Compute the gradient of f_old

            # if mvec_old.grad is not None:
            #     mvec_old.grad.zero_()  # Zero out the gradient before computing it  
            g = mvec_old.grad  # Get the gradient of mvec_old

            self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_old.detach().numpy())
            self.data_prg.append(self.dpred(mvec_old).detach().numpy())

           # Exact line search
            if ELS:
                A = self.Wd @ J             
                t = torch.dot(g,g)/torch.dot(A@g,A@g)
            else:
                t = 1.

            g_norm = torch.linalg.norm(g, ord=2)
            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            dm = t*g.flatten()  # Ensure dm is a 1D tensor
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            f_new, phid, phim = self.loss_func_L1reg(mvec_new, beta=beta,m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            if BLS:
                while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                    s *= torch.tensor(sfac)
                    mvec_new = self.project_convex_set(mvec_old - s * dm)
                    f_new, phid, phim = self.loss_func_L1reg(mvec_new,beta=beta,m_ref=m_ref) 
                    if s < torch.tensor(stol):
                        break
            mvec_old = mvec_new.detach().clone().requires_grad_()
            if print_update:
                print(f'{i + 1:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e},  f:{f_new:.1e}')

        self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_new.detach().numpy())
        self.data_prg.append(self.dpred(mvec_new).detach().numpy())
        return mvec_new

    def GaussNewton(self,mvec_init, niter, beta0, print_update=True, 
        coolingFactor=2.0, coolingRate=2, s0=1.0, sfac = 0.5,update_Wsen=False, 
        stol=1e-6, gtol=1e-3, mu=1e-4):
        mvec_old = mvec_init #.detach()
        self.error_prg = []
        self.data_prg = []
        self.mvec_prg = []
        
        m_ref = mvec_init.detach()
        f_old, phid, phim = self.loss_func_L2(mvec_old, beta=beta0, m_ref=m_ref)
        self.error_prg.append(np.array([f_old.item(), phid.item(), phim.item()]))
        self.mvec_prg.append(mvec_old.detach().numpy())
        self.data_prg.append(self.dpred(mvec_old).detach().numpy())

        for i in range(niter):
            beta = beta0* torch.tensor(1.0 / (coolingFactor ** (i // coolingRate)))
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

            if g_norm < torch.tensor(gtol):
                print(f"Inversion complete since norm of gradient is small as: {g_norm:.3e}")
                break

            s = torch.tensor(s0)
            mvec_new = self.project_convex_set(mvec_old - s * dm)
            f_new, phid, phim = self.loss_func_L2(mvec_new, beta=beta,m_ref=m_ref)
            directional_derivative = torch.dot(g.flatten(), -dm.flatten())
            while f_new >= f_old + s*torch.tensor(mu)* directional_derivative:
                s *= torch.tensor(sfac)
                mvec_new = self.project_convex_set(mvec_old - s * dm)
                f_new, phid, phim = self.loss_func_L2(mvec_new,beta=beta,m_ref=m_ref) 
                if s < torch.tensor(stol):
                    break
            mvec_old = mvec_new
            f_old = f_new
            self.error_prg.append(np.array([f_new.item(), phid.item(), phim.item()]))
            self.mvec_prg.append(mvec_new.detach().numpy())
            self.data_prg.append(self.dpred(mvec_new).detach().numpy())
            if print_update:
                print(f'{i + 1:3}, beta:{beta:.1e}, step:{s:.1e}, gradient:{g_norm:.1e},  f:{f_new:.1e}')
        return mvec_new
    
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
