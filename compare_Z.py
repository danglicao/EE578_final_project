import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import os

def main() -> None:
    ems_Z = np.loadtxt('data/openEms_Z.txt', dtype = complex)
    our_Z = np.loadtxt('data/our_Z.txt', dtype = complex)
    mat_data = scipy.io.loadmat('data/matlabimpedance.mat')
    matlab_freq = mat_data['freq'].squeeze()
    matlab_z = mat_data['z'].squeeze()

    print(ems_Z.shape)
    print(our_Z.shape)

    dt = 0.99 * 5e-3 / (3e8 * np.sqrt(np.float64(3)))
    f = np.fft.fftfreq(1800, d = dt.item())
    fig, ax = plt.subplots(figsize = (6, 4))
    ax.plot(np.fft.fftshift(f) / 1e9, np.fft.fftshift(ems_Z.real), label = "openEMS resistivity",
            color = 'b')
    ax.plot(np.fft.fftshift(f) / 1e9, np.fft.fftshift(our_Z.real), label = "our simulation resistivity",
            color = 'r')
    ax.plot(matlab_freq/1e9, matlab_z.real, label = "matlab resistivity", color = 'g')

    ax.set_ylim(0, 500)
    ax.set_xlim(0.5, 1.6)

    ax.legend()
    ax.grid(True)
    # ax.set_xlim(-c0 / lambda_L / 1e9 * 1.1, c0 / lambda_L / 1e9 * 1.1)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Resistivity(Z_real)")
    ax.set_title("Dipole resistivity")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize = (6, 4))
    ax.plot(np.fft.fftshift(f) / 1e9, np.fft.fftshift(ems_Z.imag), label = "openEMS reactivity",
            color = 'b')
    ax.plot(np.fft.fftshift(f) / 1e9, np.fft.fftshift(our_Z.imag),
            label = "our simulation reactivity",
            color = 'r')
    ax.plot(matlab_freq / 1e9, matlab_z.imag, label = "matlab reactivity", color = 'g')
    ax.plot()

    ax.set_ylim(-500, 500)
    ax.set_xlim(0.5, 1.6)

    ax.legend()
    ax.grid(True)
    # ax.set_xlim(-c0 / lambda_L / 1e9 * 1.1, c0 / lambda_L / 1e9 * 1.1)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Reactivity(Z_img)")
    ax.set_title("Dipole reactivity")
    plt.tight_layout()
    plt.show()

def see_matlab_data():

    # 读取 mat 文件
    mat_data = scipy.io.loadmat('data/matlabimpedance.mat')
    x = mat_data['freq'].squeeze()
    y = mat_data['z'].squeeze()
    print(x.shape)
    print(y.shape)


if __name__ == '__main__':
    main()
    # see_matlab_data()