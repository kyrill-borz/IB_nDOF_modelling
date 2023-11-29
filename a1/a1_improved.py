import argparse
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt


def MLKF_ndof_old(*args):
    """Return mass, damping, stiffness & force matrices for n-DOF system"""
    n = len(args) // 4
    M = np.diag(args[:n])
    L = np.diag(args[n:2*n])
    K = np.diag(args[2*n:3*n])
    F = np.array(args[3*n:])
    return M, L, K, F

def MLKF_ndof(*args):
    """Return mass, damping, stiffness & force matrices for n-DOF system"""
    n = len(args) // 4
    M = np.diag(args[:n])
    print(n)
    # Reshape the stiffness and damping components into matrices
    K_values = args[2*n:3*n]
    L_values = args[1*n:2*n]
    
    K = np.zeros((n, n))
    L = np.zeros((n, n))
    print(K_values)
# Diagonal terms for stiffness matrix
    if n > 1:
        for i in range(n):
            if i == 0:
                K[i, i] = K_values[i] + K_values[i+1]
                K[i, i+1] = -K_values[i+1]
            elif i != n-1:
                K[i, i] = K_values[i] + K_values[i+1]
                K[i, i-1] = -K_values[i]
                K[i, i+1] = -K_values[i+1]
            else:
                K[i, i] = K_values[i]
                K[i, i-1] = -K_values[i]
            
        # Diagonal terms for damping matrix
        for i in range(n):
            if i == 0:
                L[i, i] = L_values[i] + L_values[i+1]
                L[i, i+1] = -L_values[i+1]
            elif i != n-1:
                L[i, i] = L_values[i] + L_values[i+1]
                L[i, i-1] = -L_values[i]
                L[i, i+1] = -L_values[i+1]
            else:
                L[i, i] = L_values[i]
                L[i, i-1] = -L_values[i]
    else:
        K[0,0] = K_values[0]
        L[0,0] = L_values[0]
    return M, L, K, np.array(args[3*n:])

def freq_response(w_list, M, L, K, F):
    """Return complex frequency response of an n-DOF system"""
     
    outputs = np.array([np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list])
    return outputs

def time_response(t_list, M, L, K, F):

    """Return time response of system"""

    num_dofs = M.shape[0]

    def slope(t, y):
        xv = y.reshape((2, -1))
        a = np.linalg.solve(M, F - L @ xv[1, :] - K @ xv[0, :])
        s = np.concatenate((xv[1, :], a))
        return s

    solution = scipy.integrate.solve_ivp(
        fun=slope,
        t_span=(t_list[0], t_list[-1]),
        y0=np.zeros(num_dofs * 2),
        method='Radau',
        t_eval=t_list
    )

    return solution.y[:num_dofs, :].T

def last_nonzero(arr, axis, invalid_val=-1):
    """Return index of last non-zero element of an array"""
    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def plot(fig, hz, sec, M, L, K, F, show_phase=None):
    """Plot frequency and time domain responses of an n-DOF system"""
    # Generate response data
    f_response = freq_response(hz * 2 * np.pi, M, L, K, F)
    f_amplitude = np.abs(f_response)
    t_response = time_response(sec, M, L, K, F)

    # Determine suitable legends
    f_legends = (f'm{i + 1} peak {f_amplitude[m][i]:.4g} metre at {hz[m]:.4g} Hz'
                 for i, m in enumerate(np.argmax(f_amplitude, axis=0)))

    equilib = np.abs(freq_response([0], M, L, K, F))[0]  # Zero Hz
    toobig = abs(100 * (t_response - equilib) / equilib) >= 2
    lastbig = last_nonzero(toobig, axis=0, invalid_val=len(sec) - 1)

    t_legends = (f'm{i + 1} settled to 2% beyond {sec[lastbig[i]]:.4g} sec'
                 for i, _ in enumerate(t_response.T))

    # Create plot
    fig.clear()

    if show_phase is not None:
        ax = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3)
        ]
        ax[1].sharex(ax[0])
    else:
        ax = [
            fig.add_subplot(2, 1, 1),
            fig.add_subplot(2, 1, 2)
        ]

    ax[0].set_title('Amplitude of frequency domain response to sinusoidal force')
    ax[0].set_xlabel('Frequency/hertz')
    ax[0].set_ylabel('Amplitude/metre')
    ax[0].legend(ax[0].plot(hz, f_amplitude), f_legends)

    if show_phase is not None:
        p_legends = (f'm{i + 1}' for i in range(f_response.shape[1]))

        f_phases = f_response
        if show_phase == 0:
            ax[1].set_title('Phase of frequency domain response to sinusoidal force')
        else:
            f_phases /= f_response[:, show_phase - 1:show_phase]
            ax[1].set_title(
                f'Phase, relative to m{show_phase}, of frequency domain response to sinusoidal force')
        f_phases = np.degrees(np.angle(f_phases))

        ax[1].set_xlabel('Frequency/hertz')
        ax[1].set_ylabel('Phase/°')
        ax[1].legend(ax[1].plot(hz, f_phases), p_legends)

    ax[-1].set_title('Time domain response to step force')
    ax[-1].set_xlabel('Time/second')
    ax[-1].set_ylabel('Displacement/metre')
    ax[-1].legend(ax[-1].plot(sec, t_response), t_legends)

    fig.tight_layout()


def arg_parser():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''
            For an n-DOF system, show the frequency domain response to an applied sinusoidal force,
            and the time domain response to a step force.
    ''')

    ap.add_argument('args', type=float, nargs='+', help='Mass, damping, stiffness, and force values')

    ap.add_argument('--hz', type=float, nargs=2, default=(0, 5), help='Frequency range')
    ap.add_argument('--sec', type=float, default=30, help='Time limit')
    ap.add_argument('--show-phase', type=int, nargs='?', const=0,
                    help='''Show the frequency domain phase response(s).
                    If this option is given without a value then phases are shown
                    relative to the excitation.
                    If a value is given then phases are shown relative to the
                    phase of the mass with that number.
                ''')

    return ap


def main():
    """Main program"""
    # Read command line
    ap = arg_parser()
    args = ap.parse_args()

    # Generate matrices describing the system
    M, L, K, F = MLKF_ndof(*args.args)
    print(M,L,K,F)
    # Generate frequency and time arrays
    hz = np.linspace(args.hz[0], args.hz[1], 10001)
    sec = np.linspace(0, args.sec, 10001)

    # Plot results
    fig = plt.figure()
    plot(fig, hz, sec, M, L, K, F, args.show_phase)
    fig.canvas.mpl_connect('resize_event', lambda x: fig.tight_layout(pad=2.5))
    plt.show()


if __name__ == '__main__':
    main()
