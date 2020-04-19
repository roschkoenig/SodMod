import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.signal
from . import params as pr
from . import incurr as ic

#===============================================================================
# Parameter sweep plot
#===============================================================================
def sweep(Vy, I_scl, Na_scl, specs, figscale = 1):
    
    conds   = specs.conds
    cmaps   = specs.cmaps
    ctyp    = specs.ctyp
    
    # Set up plot and necessary variables
    #---------------------------------------------------------------------------
    fig     = plt.figure(constrained_layout = True, figsize=(6*figscale,1.5*len(conds)*figscale))
    fig.patch.set_facecolor('white')
    gs      = fig.add_gridspec(3,3)
    testpar = pr.params(typ = ctyp)
    Osc     = np.zeros((len(conds),len(Vy),len(Vy[0])))
    ni      = 0
    
    for vy in Vy:        # First simulation loop: Na_scl (counter: ni)
    #--------------------------------------------------------------------------
        f      = np.zeros([2,len(vy)])
        plotid = 0
  
        for ci in range(len(conds)):
            cmap   = cmaps[ci]
            ndcmap = cmap(np.linspace(0,1,len(Vy)))
            cond   = conds[ci]
            
            for i in range(len(vy)):      # Second simulation loop: I_scl (cnt: i)
            #------------------------------------------------------------------
                f[0,i] = np.min(vy[i][cond][2500:5000,0])
                f[1,i] = np.max(vy[i][cond][2500:5000,0])
            
            Osc[ci,ni,:] = f[1,:] - f[0,:]

            ax = fig.add_subplot(gs[ci, 0])
            ax.plot(np.log(I_scl), f[0,:], color=ndcmap[ni])
            ax.plot(np.log(I_scl), f[1,:], color=ndcmap[ni])
            ax.set_xlabel('ln of input current')
            ax.set_ylabel('min/max membrane potential')
            ax.set_title(conds[ci])

            if ni == 0: 
                sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin=np.log(Na_scl[0]),
                                                                             vmax=np.log(Na_scl[-1])))
                sm._A = []
                cb = plt.colorbar(sm)
                cb.set_label('ln(NaP_scale)')
            
        ni     = ni + 1

    # Plot images of two dimensional maps
    #--------------------------------------------------------------------------
    for k in range(Osc.shape[0]):
        ax = fig.add_subplot(gs[k,1:])
        mp = ax.imshow(Osc[k,:,:],aspect='auto', origin='lower', cmap ='RdGy_r',
                  extent=(np.log(I_scl[0]), np.log(I_scl[-1]), np.log(Na_scl[0]), np.log(Na_scl[-1])))
        ax.set_ylabel('ln(NaP_scale)')
        ax.set_xlabel('ln of input current')
        ax.set_title(conds[k])
        cb = fig.colorbar(mp, ax=ax)
        cb.set_label('Oscillatory amplitude')

#===============================================================================
# Bifurcation plots
#===============================================================================
def bifurcation(Vy_fwd, Vy_bwd, I_fwd, I_bwd, specs, figscale = 1, Nplots = None, direction = [0,1]):
    if Nplots == None: Nplots = len(Vy_fwd)

    # Set up plot
    #--------------------------------------------------------------------
    fig, ax = plt.subplots(len(specs.conds)+1,1, 
                           figsize=(12*figscale,4*len(specs.conds)*figscale))
    fig.patch.set_facecolor('white')
    testpar = pr.params(typ = specs.ctyp)

    plotid = 0
    for ci in range(len(specs.conds)):
        frequencies = []
        for i in range(0,len(Vy_fwd),round(len(Vy_fwd)/Nplots)):

            cond = specs.conds[ci]
            half = int(Vy_fwd[0][cond].shape[0]/2)
            if 0 in direction:
                f    = np.zeros([2,1])
                f[0] = np.min(Vy_fwd[i][cond][half:,0])
                f[1] = np.max(Vy_fwd[i][cond][half:,0])
                i_fwd    = np.multiply([1,1],np.log(I_fwd[i]))
                
                peaks = scipy.signal.find_peaks(Vy_fwd[i][cond][half:,0], height=-50)
                dur   = specs.T[half:]
                dur   = (dur[-1] - dur[0])/1000
                frequencies.append(peaks[0].shape[0]/dur)

            if 1 in direction:
                b    = np.zeros([2,1])
                b[0] = np.min(Vy_bwd[i][cond][half:,0])
                b[1] = np.max(Vy_bwd[i][cond][half:,0])
                i_bwd    = np.multiply([1,1],np.log(I_bwd[i]))

            # Do the plotting
            #------------------------------------------------------------
            if 0 in direction:  ax[plotid].scatter(i_fwd, f, color=specs.cols[ci])
            if 1 in direction:  ax[plotid].scatter(i_bwd, b, color=specs.cols[ci],        
                                                   facecolor='none')
            ax[plotid].set_ylabel('membrane potential')
        
        ax[3].scatter(np.log(I_fwd), frequencies, color=specs.cols[ci])
        
        plotid = plotid + 1

    ax[3].set_xlabel('ln of input current')
    ax[3].set_ylabel('neuronal firing frequency')
    
    return fig
    
#===============================================================================
# Phase space plots
#===============================================================================
def phasespace(Vy, I_scl, specs, states = ['m_Na', 'h_Na'], Nplots = 0, figscale=1):

    if Nplots == 0: Nplots = len(Vy)
    specs.conds = list(Vy[0].keys())

    # Set up plot
    #--------------------------------------------------------------------
    fig, ax = plt.subplots(1,len(specs.conds), 
                           figsize=(4*len(specs.conds)*figscale, 3*figscale))
    fig.patch.set_facecolor('white')
    testpar = pr.params(typ = specs.ctyp)

    for ci in range(len(specs.conds)):
        cmap   = specs.cmaps[ci]
        ndcmap = cmap(np.linspace(0,1,Nplots))

        setall = [i for i in range(0,len(Vy),int(np.floor(len(Vy)/Nplots)))]

        k = 0
        for i in setall:
            cond = specs.conds[ci]
            half = int(Vy[i][cond].shape[0] / 2)
            s0   = Vy[i][cond][-int(half/4):,testpar["snames"].index(states[0])]
            s1   = Vy[i][cond][-int(half/4):,testpar["snames"].index(states[1])]

            # Do the plotting
            #------------------------------------------------------------
            ax[ci].plot(s0, s1, color=ndcmap[k,:])
            k = k + 1
        
        ax[ci].set_title(cond)
        ax[ci].set_xlabel(states[0])
        ax[ci].set_ylabel(states[1])
        
        if 'Vm' not in states: 
            ax[ci].set_aspect('equal')
            ax[ci].set_xlim(0,1)
            ax[ci].set_ylim(0,1)
            
        # Colormap settings
        #----------------------------------------------------------------
        sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin=np.log(I_scl[0]),
                                                                     vmax=np.log(I_scl[-1])))
        sm._A = []
        cb    = plt.colorbar(sm,ax=ax[ci])
        cb.set_label('ln of input current')
    
    return fig
            
#         norm = mpl.colors.Normalize(vmin=np.log(I_scl[0]), vmax=np.log(I_scl[-1]))
#         cb   = mpl.colorbar.ColorbarBase(ax[1,ci], cmap=cmap, norm=norm,
#                                          orientation='horizontal')

    
#===============================================================================
# Time series plots
#===============================================================================
def timeseries(Vy, I_scl, specs, Nplots = 0, figscale = 1):

    if Nplots == 0: Nplots = len(Vy)
    specs.conds = list(Vy[0].keys())

    # Set up plot
    #--------------------------------------------------------------------
    fig     = plt.figure(constrained_layout = True, 
                         figsize=(12*figscale,2*len(specs.conds)*figscale))
    fig.patch.set_facecolor('white')
    gs      = fig.add_gridspec(4,1)
    ax      = fig.add_subplot(gs[0:2, 0])
    plotid  = 0

    for i in range(0,len(Vy),round(len(Vy)/Nplots)):

        for ci in range(len(specs.conds)):
            cond = specs.conds[ci]
            V    = Vy[i][cond][:,0]

            # Do the plotting
            #------------------------------------------------------------
            ax.plot(specs.T, V-ci*100, specs.cols[ci], label=specs.conds[ci])
            ax.set_title("Max Input current " + str(I_scl))
            ax.legend()

        ax = fig.add_subplot(gs[3, 0])
        ax.plot(specs.T,[ic.Id(t,specs.paradigm)*I_scl for t in specs.T])


