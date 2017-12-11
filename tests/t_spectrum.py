import numpy as np
from numpy.random import normal as randn
import tspy as ts
import matplotlib.pyplot as plt
plt.rc('text', usetex=False)


def gen_ar(t, coeffs):
    c = np.flipud(coeffs)
    p = c.size
    x = np.zeros((t+p))
    for i in range(p, t):
        x[i] = np.dot(x[i-p:i], c) + randn()
    return x[p:]


def main():
    T = 1500
    e = np.zeros((T,))

    for t in range(T):
        e[t] = randn()

    a1 = 0.7
    b1, b2 = 0.3, -0.6
    x = gen_ar(T, [a1])
    y = gen_ar(T, [b1, b2])

    w, method = 271, 'blackman'

    se = ts.spectrum(e, smooth=False, smooth_window=w, smooth_method=method)
    sx = ts.spectrum(x, smooth=False, smooth_window=w, smooth_method=method)
    sy = ts.spectrum(y, smooth=False, smooth_window=w, smooth_method=method)

    # freq = np.linspace(start=0, stop=np.pi, num=sx.size)
    freq = np.fft.rfftfreq(1024, d=1/(2*np.pi))

    tse = (1 / (2 * np.pi)) * np.ones(freq.shape)
    tsx = (1 / (2 * np.pi)) * (1 / (1 + a1 ** 2 - 2 * a1 * np.cos(freq)))
    tsy = (1 / (2 * np.pi)) * (1 / (1 + b1 ** 2 + b2 ** 2 + 2 * b1 * b2 -
                                    2 * (b1 + b2) * np.cos(freq)))

    ticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    tick_labels = ['0', r'$\frac{1}{4} \pi$', r'$\frac{2}{4} \pi$',
                   r'$\frac{3}{4} \pi$', r'$\pi$']

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

    ax[0, 0].plot(e, color='black', linewidth=0.5)
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Value')
    ax[0, 0].set_title('Gaussian white noise')
    ax[0, 1].plot(x, color='black', linewidth=0.5)
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel('Value')
    ax[0, 1].set_title('AR(1) process')
    ax[0, 2].plot(y, color='black', linewidth=0.5)
    ax[0, 2].grid(alpha=0.3)
    ax[0, 2].set_xlabel('Time')
    ax[0, 2].set_ylabel('Value')
    ax[0, 2].set_title('AR(2) process')

    ax[1, 0].plot(freq, se,  color='black', linewidth=1, label='Estimated')
    ax[1, 0].plot(freq, tse, color='red',   linewidth=2, label='Theoretical')
    ax[1, 0].legend()
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].xaxis.set_ticks(ticks)
    ax[1, 0].xaxis.set_ticklabels(tick_labels)
    ax[1, 0].set_xlabel('Frequency')
    ax[1, 0].set_ylabel('Power spectrum')
    ax[1, 0].set_title('Gaussian white noise')
    ax[1, 1].plot(freq, sx,  color='black', linewidth=1, label='Estimated')
    ax[1, 1].plot(freq, tsx, color='red',   linewidth=2, label='Theoretical')
    ax[1, 1].legend()
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].xaxis.set_ticks(ticks)
    ax[1, 1].xaxis.set_ticklabels(tick_labels)
    ax[1, 1].set_xlabel('Frequency')
    ax[1, 1].set_ylabel('Power spectrum')
    ax[1, 1].set_title('AR(1) process')
    ax[1, 2].plot(freq, sy,  color='black', linewidth=1, label='Estimated')
    ax[1, 2].plot(freq, tsy, color='red',   linewidth=2, label='Theoretical')
    ax[1, 2].legend()
    ax[1, 2].grid(alpha=0.3)
    ax[1, 2].xaxis.set_ticks(ticks)
    ax[1, 2].xaxis.set_ticklabels(tick_labels)
    ax[1, 2].set_xlabel('Frequency')
    ax[1, 2].set_ylabel('Power spectrum')
    ax[1, 2].set_title('AR(2) process')

    plt.tight_layout()
    fig.savefig('t_spectrum.pdf')
    plt.show()


if __name__ == '__main__':
    main()
