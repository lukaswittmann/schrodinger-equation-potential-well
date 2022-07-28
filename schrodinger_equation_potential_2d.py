import numpy as np
import matplotlib.pyplot as plt
import time
import random
from scipy.ndimage.interpolation import rotate
import math
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import datetime

time = datetime.datetime.now()

'''Definition der Laufvariablen'''
npoints = 500
dx = 1/npoints


psi = np.zeros(shape=(npoints), dtype=float)
d_psi = np.zeros(shape=(npoints), dtype=float)
d2_psi = np.zeros(shape=(npoints), dtype=float)
V = np.zeros(shape=(npoints), dtype=float)
l = np.arange(0,npoints, 1)


x = 0
y = 0

# Create stepfunction
n = 5 # number of minima
d = 50 # thickness of potentials
g = int((npoints-d)/n - d) # length of potentialminima, is calculated to be symmetric
h = 2000 # height of potential
t = 0 # depht of potential

for i in range(0, npoints):
    if x <= d :
        V[i] = h
        y = 0
    elif y <= g:
        V[i] = t
        y += 1
        if y == g:
            x = 0
    x += 1

#for i in range(0,npoints):
#    V[i] *= (((npoints / 2 - i) ** 2) )/ 1e7 * 300

#for i in range(0, npoints):
#    V[i] = (0.2 * 18000 * (0.01 + (dx * i - dx * npoints / 2) ** 4 - 0.2 * (dx * i - dx * npoints / 2) ** 2))


en = 0
psi[0] = 0
d2_psi[0] = psi[1]*(V[0] - en)
psi_0 = - dx

en = 0.1
delta_en_start = 0.1

nstates = 5
epsilon = 10 ** (-15)

energy = np.zeros(shape=(nstates), dtype=float)
state = np.zeros(shape=(nstates), dtype=float)

psis = np.zeros(shape=(npoints,nstates), dtype=float)

'''Loop aller zu berechnenden States'''
for istate in range(0, nstates):
    converged = 0
    counter = 0

    psi[0] = 0
    d2_psi[0] = psi[0] * (V[0] - en)

    psi[1] = 2 * psi[0] - psi_0 + d2_psi[0] * dx ** 2
    d2_psi[1] = psi[1] * (V[1] - en)

    for i in range(2, npoints - 0):
        psi[i] = 2 * psi[i - 1] - psi[i - 2] + d2_psi[i - 1] * dx ** 2
        d2_psi[i] = psi[i] * (V[i] - en)

    newval = psi[npoints - 1]
    oldval = newval

    delta_en = delta_en_start

    '''Iterationsloop'''
    while converged == 0:
        counter += 1

        '''Uptate Test-Energy und Energie-Inkrement'''
        if oldval*newval>0:
            en += delta_en
            delta_en *= 1.1
            oldval = newval
        elif oldval*newval<0:
            en -= delta_en
            delta_en *= 0.1
            en += delta_en

        '''Funktion am ersten Gitterpunkt iniziieren'''
        psi[0] = 0
        d2_psi[0] = psi[0] * (V[0] - en)

        '''Numerische Integration'''
        psi[1] = 2 * psi[0] - psi_0 + d2_psi[0] * dx ** 2
        d2_psi[1] = psi[1] * (V[1] - en)

        for i in range(2, npoints - 0):
            psi[i] = 2 * psi[i - 1] - psi[i - 2] + d2_psi[i - 1] * dx ** 2
            d2_psi[i] = psi[i] * (V[i] - en)

        newval = psi[npoints - 1]

        '''Normierung so dass das Integral ueber psi^2 = 1 ist'''
        psi2_int = 0

        for i in range(0, npoints - 0):
            psi2_int += (psi[i] ** 2) * dx

        psi2_int_sqrt = math.sqrt(psi2_int)
        psi[:] /= psi2_int_sqrt

        if (psi[npoints - 1] ** 2) < epsilon:
            converged = 1

        if (counter == 500):
            print("Error: not converged with error >", str(np.format_float_scientific((psi[npoints - 1] ** 2),precision=3)))
            converged = 1

    energy[istate] = en
    psis[:,istate] = psi[:]

    print("State " + str(istate+1) + " converged after " + str(counter) + " steps with energy " + str(en))

    '''Neuer Energiestartwert'''
    en *= 1.1


for i in range (0,nstates):
    for j in range (0,nstates):

        #plt.clf()
        plt.rcParams["figure.figsize"] = (18,6)
        fig, axs = plt.subplots(1, 3)

        # Create 2d potential, additive
        V1, V2 = np.meshgrid(V[:], V[:])
        Z = V1 + V2
        V1 = np.arange(0,npoints,1)
        V2 = np.arange(0,npoints,1)
        V1, V2 = np.meshgrid(V1, V2)

        axs[0].set_title("Potential 2D")
        im0 = axs[0].imshow(Z,cmap="plasma")
        plt.colorbar(im0, ax=axs[0])
        #axs[0].colorbar()

        # Create 2d wavefunction, multiplicative
        psi1, psi2 = np.meshgrid(psis[:,i], psis[:,j])
        Z = psi1 * psi2
        psi1 = np.arange(0,npoints,1)
        psi2 = np.arange(0,npoints,1)
        psi1, psi2 = np.meshgrid(psi1, psi2)

        axs[1].set_title("Wellenfunktion 2D")
        im1 = axs[1].imshow(Z,cmap="coolwarm")
        plt.colorbar(im1, ax=axs[1])

        axs[2].set_title("Aufenthaltswahrscheinlichkeit 2D")
        im2 = axs[2].imshow(Z**2,cmap="coolwarm")
        plt.colorbar(im2, ax=axs[2])

        fig.suptitle("Zustand ("+ str(i) + ", " + str(j) + "), Energie " + str(np.round(energy[i]+energy[j],4)))
        plt.show()
        #plt.savefig(str(time)+"state_"+str(i)+"_"+str(j)+".png",dpi=300)
        plt.pause(0.001)


# 3d Plot
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(psi1, psi2, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#plt.show()