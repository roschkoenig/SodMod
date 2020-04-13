import numpy as np

def Tgaba(s,p):
    T = np.zeros(s['GABA'].shape)
    T[np.where(s > 0)[0]] = p['GABAamp']

#-------------------------------------------------------------------------------
# GABA mediated current
#-------------------------------------------------------------------------------
def GABA(Vm, p, r, s):
    igaba = r * p['gGABA']* (Vm - p['Egaba'])
    dr    = np.multiply(p['aGABA']*Tgaba(s,p),(1-r)) - r * p['bGABA']
    return (igaba, dr)
    
