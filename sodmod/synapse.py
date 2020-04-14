import numpy as np

def Tgaba(s,p):
    T = np.zeros(s['GABA'].shape)
    T[np.where(s['GABA'] > 0)[0]] = p['GABAamp']
    return T

def GABA(Vm, p, r, s):
    ggaba = p['gGABA'] / np.sum(~np.isnan(s['GABA']))
    igaba = r * ggaba * (Vm - p['Egaba'])
    dr    = np.multiply(p['aGABA']*Tgaba(s,p),(1-r)) - r * p['bGABA']
    return (igaba, dr)
