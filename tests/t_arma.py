import numpy as np
import matplotlib.pyplot as plt
import tspy as ts


def main():
    T = 1000
    Car = [0, 0.2]
    Cma = [1]
    x = ts.gen_arma(size=T, ar=Car, ma=Cma, loc=3)

    fig, ax = plt.subplots()
    ax.plot(x[1:], color='black')
    ax.set_xlabel('Observations')
    plt.show()


if __name__ == '__main__':
    main()
