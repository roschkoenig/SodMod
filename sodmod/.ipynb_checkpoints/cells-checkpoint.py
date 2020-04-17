# Conductance models of specific cell subtypes
#===============================================================================
# Mostly based on: Destexhe et al. J Neurophys 1994 (72)
# https://pdfs.semanticscholar.org/7d54/8359042fabc4d0fcd20f8cfe98a3af9309bf.pdf
# Calcium dynamics from: https://www.physiology.org/doi/abs/10.1152/jn.1992.68.4.1384

from . import chans as ch
from . import calc as ca
from . import synapse as sy
from . import params as pr
from .incurr import Id
from scipy.integrate import odeint
import numpy as np


############################### Cell definitions ###############################

#===============================================================================
# Simulation executor
#===============================================================================
def IN(y,t,p,s=None):

    dy  = np.zeros((np.shape(p['snames'])[0],))
    sn  = p['snames']
    Vm  = y[sn.index('Vm')]


    # Evaluate intrinsic states
    #---------------------------------------------------------------------------
    l   = ch.Leak(Vm,p)
    k   = ch.K(Vm, p, y[sn.index('m_K')])
    na  = ch.Na(Vm,p, y[sn.index('m_Na')], y[sn.index('h_Na')])
    nap = ch.NaP(Vm,p,y[sn.index('m_NaP')])

    Int = l + k[0] + na[0] + nap[0]

    # Evaluate synaptic potentials
    #---------------------------------------------------------------------------

    # Calculate membrane potential
    #---------------------------------------------------------------------------
    dy[sn.index('Vm')] = (Id(t,p['paradigm'])*p['I_sc']-p['I_off'] - Int) / p['Cm']

    # Voltage sensitive gating
    #---------------------------------------------------------------------------
    dy[sn.index('m_K')]   = k[1]
    dy[sn.index('m_Na')]  = na[1]
    dy[sn.index('h_Na')]  = na[2]
    dy[sn.index('m_NaP')] = nap[1]

    return dy

def PY(y,t,p,s=None):

    dy  = np.zeros((np.shape(p['snames'])[0],))
    sn  = p['snames']
    Vm  = y[sn.index('Vm')]

    # Evaluate intrinsic states
    #---------------------------------------------------------------------------
    l   = ch.Leak(Vm,p)
    k   = ch.K(Vm, p, y[sn.index('m_K')])
    na  = ch.Na(Vm,p, y[sn.index('m_Na')], y[sn.index('h_Na')])
    nap = ch.NaP(Vm,p,y[sn.index('m_NaP')])
    km  = ch.M(Vm,p,y[sn.index('m_KM')])

    Int = l + k[0] + na[0] + nap[0] + km[0]

    # Evaluate synaptic potentials
    #---------------------------------------------------------------------------
    # gaba = sy.GABA(Vm, p, y[sn.index('rGABA')], s)

    # Syn = np.sum(gaba[0])

    # Calculate membrane potential
    #---------------------------------------------------------------------------
    dy[sn.index('Vm')] = (Id(t, p['paradigm'])*p['I_sc'] - Int) / p['Cm']

    # Voltage sensitive gating
    #---------------------------------------------------------------------------
    dy[sn.index('m_K')]   = k[1]
    dy[sn.index('m_Na')]  = na[1]
    dy[sn.index('h_Na')]  = na[2]
    dy[sn.index('m_NaP')] = nap[1]
    dy[sn.index('m_KM')]  = km[1]
    # dy[sn.index('rGABA')] = gaba[1]

    return dy

def RE(y,t,p):

    dy  = np.zeros((np.shape(p['snames'])[0],))
    sn  = p['snames']
    Vm  = y[sn.index('Vm')]

    # Evaluate intrinsic states
    #---------------------------------------------------------------------------
    l   = ch.Leak(Vm,p)
    k   = ch.K(Vm, p, y[sn.index('m_K')])
    na  = ch.Na(Vm,p, y[sn.index('m_Na')], y[sn.index('h_Na')])
    nap = ch.NaP(Vm,p,y[sn.index('m_NaP')])
    km  = ch.M(Vm,p,y[sn.index('m_KM')])
    th  = ch.Th(Vm,p,y[sn.index('m_Th')], y[sn.index('h_Th')], ca.ECa(p,y[sn.index('Ca_i')]))
    can = ch.CAN(Vm,p,y[sn.index('m_CAN')],y[sn.index('Ca_i')])
    kca = ch.KCa(Vm,p,y[sn.index('m_KCa')],y[sn.index('Ca_i')])

    Int = l + k[0] + na[0] + nap[0] + km[0] + th[0] # + can[0]  + kca[0]

    # Calculate membrane potential
    #---------------------------------------------------------------------------
    dy[sn.index('Vm')] = (Id(t, p['paradigm'])*p['I_sc'] - Int) / p['Cm']

    # Voltage sensitive gating
    #---------------------------------------------------------------------------
    dy[sn.index('m_K')]   = k[1]
    dy[sn.index('m_Na')]  = na[1]
    dy[sn.index('h_Na')]  = na[2]
    dy[sn.index('m_NaP')] = nap[1]
    dy[sn.index('m_KM')]  = km[1]
    dy[sn.index('m_CAN')] = can[1]
    dy[sn.index('m_KCa')] = kca[1]
    dy[sn.index('m_Th')]  = th[1]
    dy[sn.index('h_Th')]  = th[2]

    # Calcium dynamics
    #---------------------------------------------------------------------------
    dy[sn.index('Ca_i')] = ca.dCaI(y[sn.index('Ca_i')],th[0])

    return dy


################################## ODE Solver ##################################

#===============================================================================
# Simulation executor
#===============================================================================
def runsim(i_scl, conds, ctyp, specs, Y0 = None, paradigm='constant',
           I_off=0, NaP_scl = 1):
    
    cell = globals()[ctyp]
    Vy    = {}

    # Run simulation across conditions
    #-------------------------------------------------------------------------------  
    for ci in range(len(conds)):
        par   = pr.params(conds[ci], i_scl, ctyp, paradigm, I_off = I_off, 
                          NaP_scl = NaP_scl)
        if Y0 == None:  
            y0    = np.random.rand(len(par['snames']))
            y0[0] = specs['V0']
        else:           y0 = Y0[conds[ci]][-1,:]
        
        Vy.update({conds[ci]:odeint(cell, y0, specs['T'], args=(par,))})

    return Vy
