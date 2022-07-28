import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import math

'''Definition der Laufvariablen'''
npoints = 1000
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

# Double minimum Potenial
#for i in range(0, npoints):
#    V[i] = 5e3 * (0.01 + (dx * i - dx * npoints / 2) ** 4 - 0.2 * (dx * i - dx * npoints / 2) ** 2)

# Initialize 
en = 0
psi[0] = 0
d2_psi[0] = psi[1]*(V[0] - en)
psi_0 = - dx
en = 0.1
delta_en_start = 0.1

# Number of states to be calculated
nstates = 4

# Convergence criterium
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


plt.rcParams["figure.figsize"] = (8, 12)
fig, axs = plt.subplots(3)

for i in range(0,nstates):

    # Plot Wavefunctions
    axs[0].plot(l,psis[:, i], label="State " + str(i+1),linewidth="1")
    axs[0].set_title("Wellenfunktionen")
    axs[0].legend()

    # Plot Absolute square of wf
    axs[1].plot(l,(psis[:, i])**2, label="State " + str(i+1),linewidth="1")
    axs[1].set_title("Betragsquadrat")
    axs[1].legend()

    # Plot energy
    v_x = [0,npoints]
    v_y = [psis[0, i]+energy[i],psis[-1, i]+energy[i]]
    axs[2].plot(v_x,v_y, label="State " + str(i+1),linewidth="1")
    axs[2].set_title("Energien")

# Plot Potential in each plot for better visibility, IMPORTANT: Not to scale
axs[0].plot(l,V[:]/math.fabs(h)*1.5,'--',label="Potential",linewidth="0.7",color="gray")
axs[1].plot(l,V[:]/math.fabs(h)*1.5,'--',label="Potential",linewidth="0.7",color="gray")
axs[2].plot(l[:],V[:],'--',label="Potential",linewidth="1",color="gray")

plt.tight_layout()
plt.show()
