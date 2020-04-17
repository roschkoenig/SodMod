import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from . import params as pr

#===============================================================================
# Parameter sweep plot
#===============================================================================
def sweep(Vy, I_scl, Na_scl, ctyp, cols, figscale = 2):
    
    conds   = [k for k in Vy[0][0].keys()]        # Extract condition labels
    
    # Set up plot and necessary variables
    #---------------------------------------------------------------------------
    fig     = plt.figure(constrained_layout = True, figsize=(12*figscale,3*len(conds)*figscale))
    gs      = fig.add_gridspec(3,3)
    testpar = pr.params(typ = ctyp)
    Osc     = np.zeros((len(conds),len(Vy),len(Vy[0])))
    ni      = 0
    
    for vy in Vy:        # First simulation loop: Na_scl (counter: ni)
    #--------------------------------------------------------------------------
        f      = np.zeros([2,len(vy)])
        plotid = 0
  
        for ci in range(len(conds)):
            if cols[ci] == 'k': cmap = plt.get_cmap('Greys')
            if cols[ci] == 'b': cmap = plt.get_cmap('Blues')
            if cols[ci] == 'r': cmap = plt.get_cmap('Reds')
            ndcmap = cmap(np.linspace(0,1,len(Vy)))
            
            cond = conds[ci]
            for i in range(len(vy)):      # Second simulation loop: I_scl (cnt: i)
            #------------------------------------------------------------------
                f[0,i] = np.min(vy[i][cond][2500:5000,0])
                f[1,i] = np.max(vy[i][cond][2500:5000,0])
            
            Osc[ci,ni,:] = f[1,:] - f[0,:]

            ax = fig.add_subplot(gs[ci, 0])
            ax.plot(np.log(I_scl), f[0,:], color=ndcmap[ni])
            ax.plot(np.log(I_scl), f[1,:], color=ndcmap[ni])
            ax.set_xlim(left = -4)
            ax.set_xlabel('ln of input current')
            ax.set_ylabel('min/max membrane potential')
            ax.set_title(conds[ci])

            if ni == 0: 
                sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin=np.log(Na_scl[0]),vmax=np.log(Na_scl[-1])))
                sm._A = []
                cb = plt.colorbar(sm)
                cb.set_label('ln of scale factor for NaP conductance')
            
        ni     = ni + 1

    # Plot images of two dimensional maps
    #--------------------------------------------------------------------------
    for k in range(Osc.shape[0]):
        ax = fig.add_subplot(gs[k,1:])
        mp = ax.imshow(Osc[k,:,:],aspect='auto', origin='lower', cmap ='RdGy_r',
                  extent=(np.log(I_scl[0]), np.log(I_scl[-1]), np.log(Na_scl[0]), np.log(Na_scl[-1])))
        ax.set_xlim(left = -4)
        ax.set_ylabel('ln of scale factor for NaP conductance')
        ax.set_xlabel('ln of input current')
        ax.set_title(conds[k])
        cb = fig.colorbar(mp, ax=ax)
        cb.set_label('Oscillatory amplitude')
    
#===============================================================================
# Time series plots
#===============================================================================
def timeseries(Vy, I_scl, ctyp, Nplots = 0, paradigm='constant'):

  if Nplots == 0: Nplots = len(Vy)
  conds = list(Vy[0].keys())

  # Set up plot
  #--------------------------------------------------------------------
  fig, ax = plt.subplots(Nplots+1,1, figsize=(24, Nplots*6))
  plotid  = 0

  for i in range(0,len(Vy),round(len(Vy)/Nplots)):

    for ci in range(len(conds)):
      cond = conds[ci]
      V    = Vy[i][cond][:,0]

      # Do the plotting
      #------------------------------------------------------------
      if Nplots == 1:
        ax[plotid].plot(T, V-ci*100, cols[ci], label=conds[ci])
        ax[plotid].set_title("Max Input current " + str(I_scl))
        ax[plotid].legend()
      else:
        ax[plotid].plot(T, V-ci*100, cols[ci], label = conds[ci])
        ax[plotid].set_title("Input current" + str(I_scl[i]))
        ax[plotid].legend()

    plotid = plotid + 1
    ax[plotid].plot(T,[ic.Id(t,paradigm) for t in T])


#===============================================================================
# Phase space plots
#===============================================================================
def plot_phasespace(Vy, I_scl, ctyp, states = ['Vm', 'm_Na'], Nplots = 0):

  if Nplots == 0: Nplots = len(Vy)
  conds = list(Vy[0].keys())

  # Set up plot
  #--------------------------------------------------------------------
  fig, ax = plt.subplots(2,len(conds), figsize=(12*len(conds), 12))
  testpar = pr.params(typ = ctyp)

  for ci in range(len(conds)):
    if ci == 0: cmap = plt.get_cmap('Greys')
    if ci == 1: cmap = plt.get_cmap('Blues')
    if ci == 2: cmap = plt.get_cmap('Reds')
    ndcmap = cmap(np.linspace(0,1,Nplots))

    setall = [i for i in range(0,len(Vy),int(np.floor(len(Vy)/Nplots)))]
    set1 = np.intersect1d(np.where(np.log(I_scl) > -1)[0], np.where(np.log(I_scl) < 1)[0])

    k = 0
    for i in setall:
      cond = conds[ci]
      half = int(Vi[i][cond].shape(0) / 2)
      s0   = Vy[i][cond][half:,testpar["snames"].index(states[0])]
      s1   = Vy[i][cond][half:,testpar["snames"].index(states[1])]

      # Do the plotting
      #------------------------------------------------------------
      ax[0,ci].plot(s0, s1, cols[ci], color=ndcmap[k,:])
      ax[0,ci].set_title(cond + "Input current " + str(I_scl[i]))

      k = k + 1
    norm = mpl.colors.Normalize(vmin=np.log(I_scl[0]), vmax=np.log(I_scl[-1]))
    cb   = mpl.colorbar.ColorbarBase(ax[1,ci], cmap=cmap, norm=norm,
                                   orientation='horizontal')


#===============================================================================
# Bifurcation plots
#===============================================================================
def plot_bifurcation(Vy_fwd, Vy_bwd, I_fwd, I_bwd, ctyp, Nplots = None, direction = [0,1]):
  if Nplots == None: Nplots = len(Vy_fwd)
  conds  = list(Vy_fwd[0].keys())

  # Set up plot
  #--------------------------------------------------------------------
  fig, ax = plt.subplots(len(conds),1, figsize=(24,6*len(conds)))
  testpar = pr.params(typ = ctyp)

  plotid = 0
  for ci in range(len(conds)):
    for i in range(0,len(Vy_fwd),round(len(Vy_fwd)/Nplots)):

      cond = conds[ci]

      if 0 in direction:
        f    = np.zeros([2,1])
        f[0] = np.min(Vy_fwd[i][cond][1000:5000,0])
        f[1] = np.max(Vy_fwd[i][cond][1000:5000,0])
        i_fwd    = np.multiply([1,1],np.log(I_fwd[i]))

      if 1 in direction:
        b    = np.zeros([2,1])
        i_bwd    = np.multiply([1,1],np.log(I_bwd[i]))
        b[0] = np.min(Vy_bwd[i][cond][1000:5000,0])
        b[1] = np.max(Vy_bwd[i][cond][1000:5000,0])

      # Do the plotting
      #------------------------------------------------------------

      if 0 in direction:  ax[plotid].scatter(i_fwd, f, color=cols[ci])
      if 1 in direction:  ax[plotid].scatter(i_bwd, b, color=cols[ci], facecolor='none')

    plotid = plotid + 1
