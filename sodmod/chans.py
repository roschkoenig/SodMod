# Channel specifications
#===============================================================================
import numpy as np

#-------------------------------------------------------------------------------
# Fast Na current # From Traub et al. 1991 J Neurophys
#-------------------------------------------------------------------------------
def Na_gate(Vm, p):

    # Sodium ion-channel opening and closing rate functions
    #---------------------------------------------------------------------------
    alpha_m  = (-0.32*(Vm-p['Vt']-13.0)) / (np.exp(-(Vm-p['Vt']-13.0)/4)-1)
    beta_m   = (0.28*(Vm-p['Vt']-40.0)) / (np.exp((Vm-p['Vt']-40.0)/5)-1)

    alpha_h  = 0.128 * np.exp(-(Vm-p['Vt']-17)/18)
    beta_h   = 4 / (1 + np.exp(-(Vm-p['Vt']-40)/5))

    return alpha_m, beta_m, alpha_h, beta_h

def Na(Vm,p,m,h):

    alpha_m, beta_m, alpha_h, beta_h = Na_gate(Vm, p)

    # Steady state gating behaviour
    #---------------------------------------------------------------------------
    m_inf = 1 / (1 + np.exp(-(Vm-p['V2_m']) / p['s_m']));
    h_inf = 1 / (1 + np.exp(-(Vm-p['V2_h']) /-p['s_h']));
    t_m   = 1/(alpha_m + beta_m);
    t_h   = 1/(alpha_h + beta_h);

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_na  = (1-p['INaP']) * p['gNa'] * np.power(m, 3.0) * h * (Vm - p['ENa'])
    dm    = (m_inf - m) / t_m;
    dh    = (h_inf - h) / t_h;

    return (i_na,dm,dh)

def NaP(Vm,p,m):

    alpha_m, beta_m, alpha_h, beta_h = Na_gate(Vm,p)

    # Steady state gating behaviour
    #--------------------------------------------------------------------------
    m_inf = 1 / (1 + np.exp(-(Vm-p['V2_m']) / p['s_m']));
    t_m   = 10/(alpha_m + beta_m);

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_nap = p['INaP'] * m * (Vm - p['ENa'])
    dm    = (m_inf - m) / t_m

    return(i_nap, dm)

#-------------------------------------------------------------------------------
# Fast K current
#-------------------------------------------------------------------------------
def K(Vm,p,n):

    # Potassium ion-channel rate functions
    #---------------------------------------------------------------------------
    alpha_n = (-0.032 * (Vm - p['Vt'] -15.0)) / (np.exp(-(Vm-p['Vt']-15.0)/5) - 1.0)
    beta_n  = 0.5 * np.exp(-(Vm-p['Vt']-10)/40.0)

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_k    = p['gK'] * np.power(n, 4.0) * (Vm - p['EK'])
    dn     = (alpha_n * (1.0 - n)) - (beta_n * n)

    return (i_k,dn)

#-------------------------------------------------------------------------------
# M current - non-inactivating K current
#-------------------------------------------------------------------------------
def M(Vm,p,n):

    # Steady state gating behaviour
    #---------------------------------------------------------------------------
    n_inf = 1 / (1 + np.exp(-(Vm+35)/10));
    t_n   = p['tM']/(3.3 * np.exp((Vm+35)/20) + np.exp(-(Vm+35)/20));

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_m = p['gM']*n*(Vm-p['EK'])
    dn  = (n_inf-n)/t_n

    return(i_m, dn)

#-------------------------------------------------------------------------------
# Leak current
#-------------------------------------------------------------------------------
def Leak(Vm,p): return (p['gL'] * (Vm - p['El']))


#-------------------------------------------------------------------------------
# Low threshold Ca2+ current: I_Th
#-------------------------------------------------------------------------------
def Th(Vm,p,m,h,ECa):

    # gating steady-state values
    #---------------------------------------------------------------------------
    m_inf    = 1/(1 + np.exp(-(Vm + 52)/7.4))
    h_inf    = 1/(1 + np.exp((Vm + 80)/5))

    # time constants
    #---------------------------------------------------------------------------
    t_m      = 0.44 + 0.15 / (np.exp((Vm + 27)/10) + np.exp(-(Vm + 102)/15))
    t_h      = 22.7 + 0.27 / (np.exp((Vm + 48)/4) + np.exp(-(Vm + 406)/50))

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_th   = p['gCa'] * np.power(m, 2.0) * h * (Vm - ECa)
    dm     = -1/(t_m) * (m - m_inf)
    dh     = -1/(t_h) * (h - h_inf)

    return (i_th,dm,dh)


#-------------------------------------------------------------------------------
# Ca2 dependent K current
#-------------------------------------------------------------------------------
def KCa(Vm,p,m,Ca_i):

    # gating steady-state values
    #---------------------------------------------------------------------------
    m_inf = p['a_Ca'] * np.power(Ca_i,2.0) / (p['a_Ca'] * np.power(Ca_i,2.0) + p['b_Ca'])

    # time constants
    #---------------------------------------------------------------------------
    t_m   = 1/(p['a_Ca'] * np.power(Ca_i,2.0) + p['b_Ca'])

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_kca = p['gKCa'] * np.power(m, 2.0) * (Vm - p['EK'])
    dm    = -(1/t_m) * (m - m_inf)

    return(i_kca, dm)

#-------------------------------------------------------------------------------
# Ca dependent nonspecific cation current
#-------------------------------------------------------------------------------
def CAN(Vm,p,m,Ca_i):

    # gating steady-state values
    #---------------------------------------------------------------------------
    m_inf = p['a_CAN'] * np.power(Ca_i,2.0) / (p['a_CAN'] * np.power(Ca_i,2.0) + p['b_CAN'])

    # time constants
    #---------------------------------------------------------------------------
    t_m   = 1/(p['a_CAN'] * np.power(Ca_i,2.0) + p['b_CAN'])

    # Calculate time varying states
    #---------------------------------------------------------------------------
    i_can = p['gCAN'] * np.power(m, 2.0) * (Vm - p['ECAN'])
    dm    = -1/t_m * (m - m_inf)

    return (i_can,dm)
