# Intracellular calcium dynamics
#===============================================================================
def Ca(Ca_i,Ca_o,Th,I_T):
  
    # specific constants
    #-----------------------------------------------------------------------------
    d   = 1       # um
    F   = 96489   # C*mol-1
    R   = 8.31441 # J*mol-1 * K-1
    K_T = 0.0001  # mM/m2
    K_d = 0.0001  # mM - MM rate constants
    
    influx = -0.1/(2*F*d)*I_T
    pump   = -K_T*Ca_i/(Ca_i+K_d)
    
    dca = influx - pump
    eca = 1000 * R*p['T']/(2*F) * np.log(Ca_o/Ca_i)
    
    return (dca,eca)