# Conductance models of specific cell subtypes 
#===============================================================================
# Mostly based on: Destexhe et al. J Neurophys 1994 (72) 
# https://pdfs.semanticscholar.org/7d54/8359042fabc4d0fcd20f8cfe98a3af9309bf.pdf
# Calcium dynamics from: https://www.physiology.org/doi/abs/10.1152/jn.1992.68.4.1384

from . import chans as ch
from . import calc as ca
from .incurr import Id
import numpy as np


def IN(y,t,p):
  
    dy  = np.zeros((np.shape(p['snames_IN'])[0],))
    sn  = p['snames_IN']
    Vm  = y[sn.index('Vm')]
    

    # Evaluate intrinsic states
    #---------------------------------------------------------------------------
    l   = ch.Leak(Vm,p)
    k   = ch.K(Vm, p, y[sn.index('m_K')])
    na  = ch.Na(Vm,p, y[sn.index('m_Na')], y[sn.index('h_Na')])
    Int = l + k[0] + na[0]

    # Calculate membrane potential
    #---------------------------------------------------------------------------
    dy[sn.index('Vm')] = (Id(t) - Int) / p['Cm']

    # Voltage sensitive gating
    #---------------------------------------------------------------------------
    dy[sn.index('m_K')]   = k[1] 
    dy[sn.index('m_Na')]  = na[1]
    dy[sn.index('h_Na')]  = na[2]
    
    return dy

def cortical_hh(y, t, p):
  
    dy  = np.zeros((np.shape(p['snames'])[0],))
    sn  = p['snames']
    Vm  = y[sn.index('Vm')]
    
    # Evaluate intrinsic states
    #---------------------------------------------------------------------------
    l   = ch.Leak(Vm,p)
    k   = ch.K(Vm, p, y[sn.index('m_K')])
    na  = ch.Na(Vm,p, y[sn.index('m_Na')], y[sn.index('h_Na')])
    can = ch.CAN(Vm,p,y[sn.index('m_CAN')],y[sn.index('Ca_i')])
    kca = ch.KCa(Vm,p,y[sn.index('m_KCa')],y[sn.index('Ca_i')])
    th  = ch.Th(Vm,p,y[sn.index('m_Th')], y[sn.index('h_Th')], y[sn.index('ECa')])
    
    Int = l + k[0] + na[0] + can[0] + kca[0] + th[0]
    
    # Calculate membrane potential
    #---------------------------------------------------------------------------
    dy[sn.index('Vm')] = (Id(t) - Int) / p['Cm']
    
    # Calcium dynamics
    #---------------------------------------------------------------------------
    calc = ca.Ca(y[sn.index('Ca_i')],p['Ca_o'],p['T'],th[0])
    dy[sn.index('Ca_i')] =  calc[0]
    dy[sn.index('ECa')]  =  calc[1]
    
    # Voltage sensitive gating
    #---------------------------------------------------------------------------
    dy[sn.index('m_K')]   = k[1] 
    dy[sn.index('m_Na')]  = na[1]
    dy[sn.index('h_Na')]  = na[2]
    dy[sn.index('m_CAN')] = can[1]
    dy[sn.index('m_KCa')] = kca[1]
    dy[sn.index('m_Th')]  = th[1]
    dy[sn.index('h_Th')]  = th[2]
    
    return dy