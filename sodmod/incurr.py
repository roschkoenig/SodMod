def I_step(t):
    if   0.0  < t < 20.0:    return 0
    elif 20.0 < t < 100.0:   return 1
    return 0.0

def I_constant(t):  return 1.0

def Id(t, paradigm):
    if paradigm == 'step':  return I_step(t)
    if paradigm == 'constant': return I_constant(t)
