import numpy as np

def convmat(A, P, Q, Nx, Ny):
    NH = P*Q
    p = np.linspace(-np.floor(P/2), np.floor(P/2), P)
    q = np.linspace(-np.floor(Q/2), np.floor(Q/2), Q)

    p0 = int(1 + np.floor(Nx/2))
    q0 = int(1 + np.floor(Ny/2))

    A = np.fft.fftshift(np.fft.fftn(A))/(Nx*Ny)

    C = np.zeros((NH,NH), dtype=complex)

    for qrow in range(0, Q):
        for prow in range(0, P):
            row = qrow*P + prow
            for qcol in range(0, Q):
                for pcol in range(0, P):
                    col = qcol*P + pcol
                    pfft = int(p[prow] - p[pcol])
                    qfft = int(q[qrow] - q[qcol])
                    C[row][col] = A[p0 + pfft - 1][q0 + qfft - 1]
    return C