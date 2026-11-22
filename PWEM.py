# PWEM.py

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from convmat import convmat

# =============================================================================
# DASHBOARD
# =============================================================================

# UNIT CELL
a = 2
r = 0.15*a
er = 2

# HIGH RESOLUTION GRID
Nx = 513
Ny = Nx

# PWEM PARAMETRS
P = 11
Q = P

# NUMBER OF BLOCH VECTORS
NB = 15

# =============================================================================
# BUILD UNIT CELL
# =============================================================================
# MESHGRID
dx = a/Nx
dy = a/Ny
xa = np.linspace(1, Nx, Nx)*dx
xa = xa - np.mean(xa)
ya = np.linspace(1, Ny, Ny)*dy
ya = ya - np.mean(ya)

[Y, X] = np.meshgrid(ya, xa)

# BUILD UNIT CELL
UR = np.ones((Nx, Ny))
ER = (np.square(X) + np.square(Y)) > r*r
ER = 10 + (er - 10)*ER

# SHOW UNIT CELL
plt.pcolormesh(ER)
plt.colorbar()
plt.title('UNIT CELL')
plt.show()

# COMPUTE CONVOLUTION MATRIX
ERC = np.array(convmat(ER, P, Q, Nx, Ny), dtype=complex)
URC = np.array(convmat(UR, P, Q, Nx, Ny), dtype=complex)

# VISUALIZE CONVMAT
# plt.matshow(ERC.real)
# plt.colorbar()
# plt.title('ERC')
# plt.matshow(URC.real)
# plt.colorbar()
# plt.title('URC')
# plt.show()

# =============================================================================
# COMPUTE LIST OF BLOCH VECTORS
# =============================================================================
# COMPUTE RECIPROCAL LATTICE VECTORS
t1 = np.array([[1], [0]])
t2 = np.array([[0], [1]])
T1 = np.dot(t1, 2*np.pi/a)
T2 = np.dot(t2, 2*np.pi/a)

# COMPUTE MAIN POINTS OF LATTICE
G = np.array([[0], [0]])
X = np.dot(0.5, T1)
M = np.dot(0.5, T1) + np.dot(0.5, T2)

# COMPUTE BLOCH VECTORS LIST OX
GXx = np.linspace(G[0], X[0], NB)
XMx = np.linspace(X[0], M[0], NB)
MGx = np.linspace(M[0], G[0], int(np.floor(NB*np.sqrt(2))))

# COMPUTE BLOCH VECTORS LIST OY
GXy = np.linspace(G[1], X[1], NB)
XMy = np.linspace(X[1], M[1], NB)
MGy = np.linspace(M[1], G[1], int(np.floor(NB*np.sqrt(2))))

# COMPUTE BLOCH VECTORS LIST
bx = np.concatenate((GXx, XMx, MGx))
by = np.concatenate((GXy, XMy, MGy))

# =============================================================================
# MAIN CYCLE OF PROGRAM
# =============================================================================
# CALCULATE WAVE VECTORS
p = np.linspace(-np.floor(P/2), np.floor(P/2), P)
q = np.linspace(-np.floor(Q/2), np.floor(Q/2), Q)

# CREATING A LIST OF FREQUENCIES
WnEl = []
WnHl = []

for ii in range(0, np.size(bx)):

    # CALCULATE WAVE VECTORS
    kx = bx[ii] - 2*np.pi*p/a
    ky = by[ii] - 2*np.pi*q/a

    # CREATING MESH OF WAVE VECTORS
    [ky, kx] = np.meshgrid(ky, kx)

    # CALCULATE WAVE MATRIX OF WAVE VECTORS
    KX = np.diagflat(kx)
    KY = np.diagflat(ky)

    # BUILD EIG-VALUE PROBLEM E-MODE
    AE = np.dot(np.dot(KX, np.linalg.inv(URC)), KX) + np.dot(np.dot(KY, np.linalg.inv(URC)), KY)
    BE = ERC

    # BUILD EIG-VALUE PROBLEM H-MODE
    AH = np.dot(np.dot(KX, np.linalg.inv(ERC)), KX) + np.dot(np.dot(KY, np.linalg.inv(ERC)), KY)
    BH = URC

    # CALCULATE EIG-VALUE
    [DEval, DEvec] = linalg.eig(AE, BE)
    [DHval, DHvec] = linalg.eig(AH, BH)

    # SORTING AND NORMALIZATION FREQUENCIES
    WnEl.append(np.sort((a*np.sqrt(DEval)/(2*np.pi))))
    WnHl.append(np.sort((a*np.sqrt(DHval)/(2*np.pi))))

# CREATE AN ARRAY OF FREQUENCIES
WnE = np.array(WnEl)
WnH = np.array(WnHl)

# =============================================================================
# SHOW BRILLOUIN ZONE
# =============================================================================
# SHOW E-MODE AND H-MODE
fig, ax = plt.subplots()
plt.plot(np.real(WnE), 'r')
ax.set(xlim=[0, 2*NB + int(np.floor(NB*np.sqrt(2))) - 1], ylim=[0, 0.6])
plt.xticks([0, NB, 2*NB, 2*NB + int(np.floor(NB*np.sqrt(2))) - 1], [r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
plt.xlabel(r'Bloch wave vector $\vec\beta$')
plt.ylabel(r'Normalized Frequency $\omega$a/2$\pi$c')
plt.grid(axis='x')
plt.title('Band Diagram of a Square Lattice')
plt.show()

fig2, ax = plt.subplots()
plt.plot(np.real(WnH), 'b')
ax.set(xlim=[0, 2*NB + int(np.floor(NB*np.sqrt(2))) - 1], ylim=[0, 0.6])
plt.xticks([0, NB, 2*NB, 2*NB + int(np.floor(NB*np.sqrt(2))) - 1], [r'$\Gamma$', 'X', 'M', r'$\Gamma$'])
plt.xlabel(r'Bloch wave vector $\vec\beta$')
plt.ylabel(r'Normalized Frequency $\omega$a/2$\pi$c')
plt.grid(axis='x')
plt.title('Band Diagram of a Square Lattice')
plt.show()
