from scipy.optimize import curve_fit
from .chans import Na_gate
import numpy as np
from matplotlib import pyplot as plt

#         # input resistance 2000 MOhms, surface area 1000 um2
#         # Taken from Destexhe A, [...] Steriade M (1994) J Neurophys
#         # Indices of states - XX
#         #-------------------------------------------------------------------------------
#         snames = [None]*10;
#         snames[0] = 'Vm'           # Membrane voltage
#         snames[1] = 'Ca_i'         # Intracellular calcium
#         snames[2] = 'ECa'          # Calcium reversal potential
#         snames[3] = 'm_K'          # Potassium gate
#         snames[4] = 'm_Na'         # sodium activation (?) gate
#         snames[5] = 'h_Na'         # sodium fast inactivation (?) gate
#         snames[6] = 'm_CAN'        # nonspecific cation activation gate
#         snames[7] = 'm_KCa'        # calcium dependent potassium activation gate
#         snames[8] = 'm_Th'         #
#         snames[9] = 'h_Th'         #
#         p['snames_XX'] = snames

#         p['gCa']  = 1.75            # calcium conductance (mS/cm^2)
#         p['gKCa'] = 10              # calcium dependent potassium conductance (mS/cm^2)
#         p['gCAN'] = 0.25            # Ca dependent nonspecific cation conductance
#

# p['ECAN'] = -20             # Ca dependent nonspecific cation rev pot (mV)

 # Intracellular calcium dynamics parameters
#         #-------------------------------------------------------------------------------
#         p['Ca_o'] = 2               # Extracellular calcium (mM)
#         p['T']    = 309.15          # Temperature (K)

#         # Other special parameters
#         #-------------------------------------------------------------------------------
#         p['a_Ca']  = 48            # ms-1 mM-2
#         p['b_Ca']  = 0.03          # ms-1
#         p['a_CAN'] = 20            # ms-1 mM-2
#         p['b_CAN'] = 0.002         # ms-1


def initialise(p, V0=-80, style='zeros'):
    if style == 'random': y0 = np.random.rand(len(p['snames']))
    if style == 'zeros': y0 = np.zeros(len(p['snames']))
    return y0


def params(cond = 'WT37', I_scale = 1, typ = 'IN', paradigm = 'step', I_off = 0, NaP_scl = 0.1):

    p = {}

    # Fixed values
    #-------------------------------------------------------------------------------
    p['Cm']    = 1               # membrance capacitance (uF/cm^2)
    p['EK']    = -90.0           # Potassium potential (mV)
    p['ENa']   = 55.0            # Sodium potential (mV)
    p['Vt']    = -63             # Firing threshold
    p['I_sc']  = I_scale         # Input scaling - This should really be changed
    p['I_off'] = I_off           # negative offsetting of input current
    p['NaP_scl'] = NaP_scl       # scaling up and down effects of persistent sodium

    # Synaptic parameters (GABA-A transmitter)
    #-------------------------------------------------------------------------------
    p['Egaba'] = -80           # GABA reversal potential (Desthexe et al 1994)
    p['aGABA'] = 0.53          # ms^(-1) mM^(-1)
    p['bGABA'] = 0.184         # ms^(-1)
    p['gGABA'] = 1             # µS
    p['GABAamp'] = 1           # mM
    p['GABAdur'] = 1           # ms


    # Type-specific parameters
    #-------------------------------------------------------------------------------
    if typ == 'IN':

        # Indices of states - IN
        #-------------------------------------------------------------------------------
        snames = [None]*5;
        snames[0] = 'Vm'           # Membrane voltage
        snames[1] = 'm_K'          # Potassium gate
        snames[2] = 'm_Na'         # sodium activation gate
        snames[3] = 'h_Na'         # sodium fast inactivation gate
        snames[4] = 'm_NaP'        # Persistent sodium current gate
        p['snames'] = snames

        # Conductances and membrane capacitance
        #-------------------------------------------------------------------------------
        p['A']    = 0.00014         # surface area (29,000um^2 in cm^2)
        p['gK']   = 10              # potassium conductance (mS/cm^2)
        p['gNa']  = 50.0            # sodium conductance (mS/cm^2)
        p['gL']   = 0.1             # leak conductance (mS/cm^2)

        # Reversal potentials
        #-------------------------------------------------------------------------------
        p['El']   = -70.0           # Leak potential (mV)

    # Type-specific parameters
    #-------------------------------------------------------------------------------
    if typ == 'PY':

        # Indices of states - PY
        #-------------------------------------------------------------------------------
        snames = [None]*6;
        snames[0] = 'Vm'           # Membrane voltage
        snames[1] = 'm_K'          # Potassium gate
        snames[2] = 'm_Na'         # sodium activation gate
        snames[3] = 'h_Na'         # sodium fast inactivation gate
        snames[4] = 'm_NaP'        # Persistent sodium current gate
        snames[5] = 'm_KM'         # M-type potassium activation gate
        p['snames'] = snames

        # Conductances and membrane capacitance
        #-------------------------------------------------------------------------------
        p['A']    = 0.00029         # surface area (29,000um^2 in cm^2)
        p['gK']   = 10.0            # potassium conductance (mS/cm^2)
        p['gNa']  = 50.0            # sodium conductance (mS/cm^2)
        p['gM']   = 0.07            # M type potassium conductance (mS/cm^2)
        p['gL']   = 0.1             # leak conductance (mS/cm^2)

        # Reversal potentials
        #-------------------------------------------------------------------------------
        p['El']   = -70.0           # Leak potential (mV)

        # Special parameters
        #-------------------------------------------------------------------------------
        p['tM']   = 4               # M-type gating max tau (s)

    if typ == 'RE':

        # Indices of states - RE
        #-------------------------------------------------------------------------------
        snames = [None]*11;
        snames[0] = 'Vm'           # Membrane voltage
        snames[1] = 'm_K'          # Potassium gate
        snames[2] = 'm_Na'         # sodium activation gate
        snames[3] = 'h_Na'         # sodium fast inactivation gate
        snames[4] = 'm_NaP'        # Persistent sodium current gate
        snames[5] = 'm_KM'         # M-type potassium activation gate
        snames[6] = 'm_CAN'        # Calcium dependent nonspecific cation current
        snames[7] = 'm_KCa'        # Calcium dependent K current
        snames[8] = 'm_Th'         # Low threshold Ca current activation
        snames[9] = 'h_Th'         # Low threshold Ca current inactivation
        snames[10] = 'Ca_i'        # Intracellular calcium concentration
        p['snames'] = snames

        # Conductances
        #-------------------------------------------------------------------------------
        p['A']    = 0.00029         # surface area (29,000um^2 in cm^2)
        p['gK']   = 10.0            # potassium conductance (mS/cm^2)
        p['gNa']  = 50.0            # sodium conductance (mS/cm^2)
        p['gM']   = 0.07            # M type potassium conductance (mS/cm^2)
        p['gL']   = 0.1             # leak conductance (mS/cm^2)
        p['gCa']  = 1.75            # calcium conductance (mS/cm^2)
        p['gKCa'] = 10              # calcium dependent potassium conductance (mS/cm^2)
        p['gCAN'] = 0.25            # Ca dependent nonspecific cation conductance
        p['ECAN'] = -20             # Ca dependent nonspecific cation rev pot (mV)

        # Reversal potentials
        #-------------------------------------------------------------------------------
        p['El']   = -70.0           # Leak potential (mV)

        # Special parameters
        #-------------------------------------------------------------------------------
        p['tM']   = 4               # M-type gating max tau (s)

        # Intracellular calcium dynamics parameters
        #-------------------------------------------------------------------------------
        p['Ca_o'] = 2               # Extracellular calcium (mM)
        p['T']    = 309.15          # Temperature (K)

        # Other special parameters
        #-------------------------------------------------------------------------------
        p['a_Ca']  = 48            # ms-1 mM-2
        p['b_Ca']  = 0.03          # ms-1
        p['a_CAN'] = 20            # ms-1 mM-2
        p['b_CAN'] = 0.002         # ms-1


    # Fit Boltzman description of Na steady state parameters
    #-------------------------------------------------------------------------------
    V2_m, s_m, V2_h, s_h = gatefit(p)
    p['V2_m'] = V2_m
    p['s_m']  = s_m
    p['V2_h'] = V2_h
    p['s_h']  = s_h

    # Pack up experimental simulation paradigm
    p['paradigm'] = paradigm

    return exvals(p, cond)


#===================================================================================
# Fit Boltzman equation to steady-state gating parameters
#===================================================================================
def gatefit(p):

    # Define steady state formulation from gating parameters
    #-------------------------------------------------------------------------------
    def psim(Vm, p):
        alpha_m, beta_m, alpha_h, beta_h = Na_gate(Vm, p)

        m_inf = alpha_m / (alpha_m + beta_m);
        h_inf = alpha_h / (alpha_h + beta_h);

        t_m = 1/(alpha_m + beta_m);
        t_h = 1/(alpha_h + beta_h);

        return (m_inf, t_m, h_inf, t_h)


    # Simulate steady state gating parameters over relevant range
    #-------------------------------------------------------------------------------
    vrange = range(-80,20,1)
    vals   = np.ndarray( (5, len(vrange)))
    i      = 0

    for V in vrange:
        val       = psim(V, p)
        vals[0,i] = V;
        vals[1,i] = val[0];
        vals[2,i] = val[1];
        vals[3,i] = val[2];
        vals[4,i] = val[3];
        i = i+1;

    # Remove NaN values by interpolating
    #-------------------------------------------------------------------------------
    for i in range(1,5):
        nanlocs = np.where(np.isnan(vals[i,]))[0]
        for n in nanlocs:
            vals[i,n] = np.mean(np.array((vals[i,n-1], vals[i,n+1])))

    # Fit gating parameters
    #-------------------------------------------------------------------------------
    def F_m(V,V2_m,s_m):return 1 / (1 + np.exp(-(np.divide(V-V2_m,s_m))))
    def F_h(V,V2_h,s_h):return 1 / (1 + np.exp((np.divide(V-V2_h,s_h))))

    p0          = [-12,3]
    mfit, mcvar = curve_fit(F_m, vals[0,], vals[1,], p0)
    hfit, hcvar = curve_fit(F_h, vals[0,], vals[3,], p0)

    # Return V2_m, sm, V2_h, sh
    #-------------------------------------------------------------------------------
    return mfit[0], mfit[1], hfit[0], hfit[1]


#===================================================================================
# Empirical parameter adjustments
#===================================================================================
def exvals(p, cond = 'WT37'):
    e = []

    e.append({'name' : 'WT32',
             'temp' : 32 + 273,
             'V2_m' : -13.3,
             's_m'  : 3.9,
             'V2_h' : -48.1,
             's_h'  : -4.4,
             'Frec' : 2.16,
             'rNaP' : 0.0164 }
            )

    e.append({'name' : 'WT37',
             'temp' : 37 + 273,
             'V2_m' : -16.4,
             's_m'  : 4.5,
             'V2_h' : -50.2,
             's_h'  : -4.4,
             'Frec' : 1.28,
             'rNaP' : 0.0191 }
            )

    e.append({'name' : 'AS32',
             'temp' : 32 + 273,
             'V2_m' : -16.0,
             's_m'  : 4.0,
             'V2_h' : -50.7,
             's_h'  : -4.2,
             'Frec' : 2.05,
             'rNaP' : 0.0184 }
            )

    e.append({'name' : 'AS37',
             'temp' : 37 + 273,
             'V2_m' : -11.0,
             's_m'  : 4.2,
             'V2_h' : -43.5,
             's_h'  : -4.5,
             'Frec' : 1.06,
             'rNaP' : 0.0194 }
            )

    e.append({'name' : 'TI32',
             'temp' : 32 + 273,
             'V2_m' : -18.5,
             's_m'  : 3.8,
             'V2_h' : -50.3,
             's_h'  : -3.9,
             'Frec' : 2.89,
             'rNaP' : 0.0466 }
            )

    e.append({'name' : 'TI37',
             'temp' : 37 + 273,
             'V2_m' : -16.1,
             's_m'  : 4.0,
             'V2_h' : -50.8,
             's_h'  : -4.0,
             'Frec' : 1.47,
             'rNaP' : 0.0659 }
            )
    
    e.append({'name' : 'TI-AS-32',
             'temp' : 32 + 273,
             'V2_m' : -24.74,
             's_m'  : 3.9,
             'V2_h' : -59.3,
             's_h'  : -4.1,
             'Frec' : 5.0,
             'rNaP' : 0.0560 }
            )

    e.append({'name' : 'TI-AS-37',
             'temp' : 37 + 273,
             'V2_m' : -20.21,
             's_m'  : 4.21,
             'V2_h' : -51.8,
             's_h'  : -4.1,
             'Frec' : 1.28,
             'rNaP' : 0.0713 }
            )

    # Normalise to wildtype at 37 degrees and calculate differences
    #--------------------------------------------------------------------------
    norm = list(filter(lambda e: e['name'] == 'WT37', e))[0]
    curr = list(filter(lambda e: e['name'] == cond, e))[0]

    dV2_m = curr['V2_m'] - norm['V2_m']
    m_z   = (0.0863 * norm['temp']) / p['s_m']  # expected value at body temp
    m_z   = curr['s_m']/norm['s_m'] * m_z       # adjustment by s_m ratios
    s_m   = (0.0863 * curr['temp'] / m_z)       # readjust by actual temp

    dV2_h = curr['V2_h'] - norm['V2_h']
    h_z   = (0.0863 * norm['temp']) / p['s_h']  # expected value at body temp
    h_z   = curr['s_h']/norm['s_h'] * h_z       # adjustment by s_m ratios
    s_h   = (0.0863 * curr['temp'] / h_z)       # readjust by actual temp

    rNaP  = curr['rNaP']

    # Pack into parameter dictionariy
    #--------------------------------------------------------------------------
    p['V2_m'] = p['V2_m'] + dV2_m
    p['V2_h'] = p['V2_h'] + dV2_h
    p['s_m']  = s_m
    p['s_h']  = s_h
    p['rNaP'] = rNaP * p['NaP_scl']

    return p


def plot_gating(conds = ['WT37', 'AS37', 'TI37']):
    def F_m(V,V2_m,s_m):return 1 / (1 + np.exp(-(np.divide(V-V2_m,s_m))))
    def F_h(V,V2_h,s_h):return 1 / (1 + np.exp((np.divide(V-V2_h,s_h))))

    fig, ax = plt.subplots(1,1, figsize=(12, 8))
    V = np.array(range(-80,20,1))

    cols  = ['k','b','r']

    for ci in range(len(conds)):
        p = params(conds[ci])
        ax.plot(V, F_m(V, p['V2_m'], p['s_m']), cols[ci])
        ax.plot(V, F_h(V, p['V2_h'], p['s_h']), cols[ci])
