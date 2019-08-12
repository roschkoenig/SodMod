import numpy as np

# Intracellular calcium dynamics
#===============================================================================
def dCaI(Ca_i, I_Th):
    K_T = 0.0001  # mM/m2
    K_d = 0.0001  # mM - MM rate constants
    F   = 96489   # C*mol-1
    d   = 1       # um
    
    influx = -0.1*I_Th/(2*F*d)
    pump   = -K_T*Ca_i/(Ca_i+K_d)
    dca    = influx - pump

    return dca 
    
def ECa(p,Ca_i):
  
    # specific constants
    #----------------------------------------------------------------------------- 
    R   = 8.31441 # J*mol-1 * K-1
    F   = 96489   # C*mol-1
    eca = 1000 * R*p['T']/(2*F) * np.log(p['Ca_o']/Ca_i)
    
    return eca