import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm
from physics_functions import fdtd_functions


def main():
    data_save_path = 'data/our_Z.txt'
    # Normalized
    # epsilon0 = cp.float64(1.0)
    # mu0 = cp.float64(1.0)
    # c0 = cp.float64(1.0)
    # sigma_max = 200

    # Not Normalized
    epsilon0 = cp.float64(8.85e-12)
    mu0 = cp.float64(4 * cp.pi * 1e-7)
    c0 = 1 / cp.sqrt(epsilon0 * mu0)
    # sigma_max = cp.float64(0.9e12) # without fixing J
    sigma_max = cp.float64(1e11)  # fixed J

    pml_thickness = 16
    # sigma_max = cp.float64(1e15)  # without fixing J
    # sigma_max = -(3 + 1) / 4 * (c0 / pml_thickness) * np.log(1e-10)
    # lambda_0 = cp.float64(600e-3)
    # lambda_U = cp.float64(450e-3)
    # lambda_L = cp.float64(350e-3)

    lambda_0 = cp.float64(300e-3)
    lambda_U = lambda_0 * 1.6
    lambda_L = lambda_0 * 0.7

    # print(c0/lambda_0/1e9)

    dx = dy = dz = cp.float64(5e-3)  # m, spatial step size
    dt = 0.99 * dx / (c0 * cp.sqrt(cp.float64(3)))

    grid_m = 200e-3
    x_min, x_max = -1 * grid_m, grid_m
    y_min, y_max = -1 * grid_m, grid_m
    z_min, z_max = -1 * grid_m, grid_m

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = 1800

    x_src, y_src, z_src = 0, 0, 0
    tmp = 1
    # x_prob, y_prob, z_prob = 0, 0, 0 # for dipole impedance
    # x_prob, y_prob, z_prob = 0 + tmp * dx, 0 + tmp * dy, 0 + tmp * dz
    x_prob, y_prob, z_prob = x_min + (pml_thickness + tmp) * dx, y_min + (
                pml_thickness + tmp) * dy, z_min + (
                                         pml_thickness + tmp) * dz  # for freespace impedance

    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))
    i_z_src = int(round((z_src - z_min) / dz))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_z_prob = int(round((z_prob - z_min) / dz))

    # Antenna hyper parameters
    L = cp.float64(lambda_0 / (2))
    L_rel = L // dz
    half_len = L_rel // 2
    i_z_dipole_start = i_z_src - half_len
    i_z_dipole_end = i_z_src + half_len + 1

    Ex_record = cp.zeros(nt, dtype = cp.float64)
    Ey_record = cp.zeros(nt, dtype = cp.float64)
    Ez_record = cp.zeros(nt, dtype = cp.float64)
    Hx_record = cp.zeros(nt, dtype = cp.float64)
    Hy_record = cp.zeros(nt, dtype = cp.float64)
    Hz_record = cp.zeros(nt, dtype = cp.float64)
    Ez_gap = cp.zeros(nt, dtype = cp.float64)
    Jxy_gap = cp.zeros(nt, dtype = cp.float64)

    omega_0 = 2 * cp.pi * c0 / lambda_0
    # sigma = (2 / omega_0) * (lambda_0 / (lambda_U - lambda_L))
    sigma = (2 / omega_0) * lambda_0

    Ex = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float64)
    Ey = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float64)
    Ez = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float64)
    Hx = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float64)
    Hy = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float64)
    Hz = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float64)
    Bx = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float64)
    By = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float64)
    Bz = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float64)
    Dx = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float64)
    Dy = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float64)
    Dz = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float64)
    Bx_old = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float64)
    By_old = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float64)
    Bz_old = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float64)
    Dx_old = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float64)
    Dy_old = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float64)
    Dz_old = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float64)
    epsilon = cp.ones((Nx, Ny, Nz), dtype = cp.float64) * epsilon0
    mu = cp.ones((Nx, Ny, Nz), dtype = cp.float64) * mu0
    excitation = cp.zeros((Nx, Ny, Nz), dtype = cp.float64)
    Ez_excitation = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float64)

    sigma_x_vec, sigma_y_vec, sigma_z_vec = fdtd_functions.pml_profile(sigma_max, pml_thickness, Nx,
                                                                       Ny, Nz)
    sigma_x_3d, sigma_y_3d, sigma_z_3d = cp.meshgrid(sigma_x_vec, sigma_y_vec, sigma_z_vec,
                                                     indexing = 'ij')

    Ex_time_record = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float64)
    Ey_time_record = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float64)
    Ez_time_record = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float64)
    Hx_time_record = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float64)
    Hy_time_record = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float64)
    Hz_time_record = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float64)

    params = fdtd_functions.fdtd_param_alignment_with_pec(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                                                 sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu,
                                                 i_x_src, i_y_src, i_z_src, i_z_dipole_start,
                                                 i_z_dipole_end)


    # main loop
    for n in tqdm(range(nt)):
        # add source
        Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old = fdtd_functions.update_equations_with_pec(
            Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
            Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
            params,
            dt, dx, dy, dz)
        excitation[i_x_src][i_y_src][i_z_src] = dt * fdtd_functions.gprmax_gaussian_source(n, dt, omega_0)
        Ez_excitation[:, :, 1:-1] = 0.5 * (excitation[:, :, :-1] + excitation[:, :, 1:])
        Ez += Ez_excitation
        # if n == 100:
        #     print(cp.unique(excitation))
        #     print(cp.unique(Ez_excitation))
        #     nonzero_indices_excitation = cp.nonzero(excitation)
        #     nonzero_indices_Ez_excitation = cp.nonzero(Ez_excitation)
        #     print("Excitation non-zero indices:", nonzero_indices_excitation)
        #     print("Ez_excitation non-zero indices:", nonzero_indices_Ez_excitation)

        # Ez[i_x_src][i_y_src][i_z_src] += dt * params.epsilon_Ez[i_x_src][i_y_src][i_z_src] * fdtd_functions.gaussian_source(n, dt, sigma, omega_0)*1e4
        # Ez[i_x_src][i_y_src][i_z_src+1] += dt * fdtd_functions.gprmax_gaussian_source(n, dt, omega_0)
        # Ez[i_x_src][i_y_src][i_z_src] += fdtd_functions.gaussian_source(n, dt, sigma, omega_0)

        Ex_record[n] = Ex[i_x_prob, i_y_prob, i_z_prob]
        Ey_record[n] = Ey[i_x_prob, i_y_prob, i_z_prob]
        Ez_record[n] = Ez[i_x_prob, i_y_prob, i_z_prob]
        Hx_record[n] = Hx[i_x_prob, i_y_prob, i_z_prob]
        Hy_record[n] = Hy[i_x_prob, i_y_prob, i_z_prob]
        Hz_record[n] = Hz[i_x_prob, i_y_prob, i_z_prob]

        Ex_record[n] = (Ex[i_x_prob, i_y_prob, i_z_prob] + Ex[i_x_prob + 1, i_y_prob, i_z_prob]) / 2
        Ey_record[n] = (Ey[i_x_prob, i_y_prob, i_z_prob] + Ey[i_x_prob, i_y_prob + 1, i_z_prob]) / 2
        Ez_record[n] = (Ez[i_x_prob, i_y_prob, i_z_prob] + Ez[i_x_prob, i_y_prob, i_z_prob + 1]) / 2
        Hx_record[n] = (Hx[i_x_prob, i_y_prob, i_z_prob] + Hx[
            i_x_prob, i_y_prob + 1, i_z_prob + 1]) / 2
        Hy_record[n] = (Hy[i_x_prob, i_y_prob, i_z_prob] + Hy[
            i_x_prob + 1, i_y_prob, i_z_prob + 1]) / 2
        Hz_record[n] = (Hz[i_x_prob, i_y_prob, i_z_prob] + Hz[
            i_x_prob + 1, i_y_prob + 1, i_z_prob]) / 2
        # Ez_gap[n] = (Ez[i_x_src, i_y_src, i_z_src] + Ez[i_x_src, i_y_src, i_z_src+1])* dz
        Ez_gap[n] = (Ez[i_x_src, i_y_src, i_z_src]) * dz
        # Jxy_gap[n] = fdtd_functions.gaussian_source(n, dt, sigma, omega_0) * dx * dy * dt
        Jxy_gap[n] = fdtd_functions.gprmax_gaussian_source(n, dt, omega_0) * dx * dy * dt

        if n == 215:
        # if n == int(nt / 2) + 20:
            Ex_time_record = cp.copy(Ex)
            Ey_time_record = cp.copy(Ey)
            Ez_time_record = cp.copy(Ez)
            Hx_time_record = cp.copy(Hx)
            Hy_time_record = cp.copy(Hy)
            Hz_time_record = cp.copy(Hz)
            # print(Ez_excitation[i_x_src][i_y_src][i_z_src - 1:i_z_src + 2])

    Ex_time_record_cpu = Ex_time_record.get()
    Ey_time_record_cpu = Ey_time_record.get()
    Ez_time_record_cpu = Ez_time_record.get()
    Hx_time_record_cpu = Hx_time_record.get()
    Hy_time_record_cpu = Hy_time_record.get()
    Hz_time_record_cpu = Hz_time_record.get()

    gpu_vars = {
        'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
        'Bx': Bx, 'By': By, 'Bz': Bz,
        'Dx_old': Dx_old, 'Dy_old': Dy_old, 'Dz_old': Dz_old,
        'Bx_old': Bx_old, 'By_old': By_old, 'Bz_old': Bz_old,
    }

    Ex_record_cpu = Ex_record.get()
    Ey_record_cpu = Ey_record.get()
    Ez_record_cpu = Ez_record.get()
    Hx_record_cpu = Hx_record.get()
    Hy_record_cpu = Hy_record.get()
    Hz_record_cpu = Hz_record.get()

    # plotting
    # 2D Field
    plot_final_fields(Ex_time_record_cpu, Ey_time_record_cpu, Ez_time_record_cpu,
                      Hx_time_record_cpu, Hy_time_record_cpu, Hz_time_record_cpu, Nx, Ny, Nz)
    # figure_3D_array_slices(Ex_time_record_cpu, Ey_time_record_cpu, Ez_time_record_cpu, cmap='bwr')
    # figure_3D_array_slices(Hx_time_record_cpu, Hy_time_record_cpu, Hz_time_record_cpu, cmap='bwr')
    plt.show()

    t = np.arange(nt) * dt.get()
    plot_probe_fields(t, Ex_record_cpu, Ey_record_cpu, Ez_record_cpu, Hx_record_cpu, Hy_record_cpu,
                      Hz_record_cpu)

    # Freespace Impedance
    E = np.sqrt(
        Ex_record_cpu ** 2 + Ey_record_cpu ** 2 + Ez_record_cpu ** 2)  # np.sqrt(Ex_record_cpu**2 + Ey_record_cpu**2 + Ez_record_cpu**2)
    H = np.sqrt(
        Hx_record_cpu ** 2 + Hy_record_cpu ** 2 + Hz_record_cpu ** 2)  # np.sqrt(Hx_record_cpu**2 + Hy_record_cpu**2 + Hz_record_cpu**2)
    E_f = np.fft.fft(E)
    H_f = np.fft.fft(H)

    Ez_gap_f = np.fft.fft(Ez_gap.get())
    Jxy_gap_f = np.fft.fft(Jxy_gap.get())
    # print(Ez_gap.get())
    # print(Jxy_gap.get())

    # half = nt // 2
    # freqs_hz = freqs_hz[:half]
    # E_f = E_f[:half]
    # H_f = H_f[:half]
    # Ez_gap_f = Ez_gap_f[:half]
    # Jxy_gap_f = Jxy_gap_f[:half]
    freqs_hz = np.fft.fftfreq(nt, d = dt.item())
    print(np.fft.fftshift(freqs_hz)[0])
    print(np.fft.fftshift(freqs_hz)[-1])
    positive = freqs_hz > 0
    # freqs_hz = freqs_hz
    # freqs_hz = freqs_hz[positive]
    # E_f = E_f[positive]
    # H_f = H_f[positive]
    # Ez_gap_f = Ez_gap_f[positive]
    # Jxy_gap_f = Jxy_gap_f[positive]

    Z_freespace = np.divide(abs(E_f), abs(H_f), out = np.zeros_like(abs(E_f)),
                            where = (abs(H_f) != 0))
    Z_dipole = np.divide(Ez_gap_f, Jxy_gap_f, out = np.zeros_like(Ez_gap_f),
                         where = (abs(Jxy_gap_f) != 0))
    # print(Z_dipole.real)
    # print(Z_dipole.imag)

    # plt.figure(figsize = (6, 4))
    # plt.plot(freqs_hz/1e9, Z_freespace, label = "|E|/|H|")
    # plt.vlines(c0.get()/lambda_0/1e9, -2, 1e4, colors = "k")
    # plt.legend()
    # plt.grid(True)
    # plt.xlim(c0/lambda_U/1e9, c0/lambda_L/1e9)
    # plt.ylim(-2, 500)
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("Impedance")
    # plt.title("Freespace Impedance")
    # plt.tight_layout()
    # plt.show()

    # print(Z_dipole)
    # fig, ax1 = plt.subplots(figsize = (6, 4))
    # ax1.plot(np.fft.fftshift(freqs_hz) / 1e9, np.fft.fftshift(Z_dipole.real), label = "real",
    #          color = 'b')
    # ax1.vlines(c0.get() / lambda_0 / 1e9, -1e4, 1e4, colors = "k")
    # ax1.vlines(c0.get() / lambda_L / 1e9, -1e4, 1e4, colors = "k", linestyle = "dotted")
    # ax1.vlines(c0.get() / lambda_U / 1e9, -1e4, 1e4, colors = "k", linestyle = "dotted")
    # plt.ylim(0, 500)
    # ax1.tick_params('y', colors = 'b')
    # ax2 = ax1.twinx()
    # ax2.plot(np.fft.fftshift(freqs_hz) / 1e9, np.fft.fftshift(Z_dipole.imag), label = "imag",
    #          color = 'r')
    # ax2.tick_params('y', colors = 'r')
    # plt.ylim(-200, 200)
    # fig.legend()
    # plt.grid(True)
    # # plt.xlim(c0/lambda_U/1e9/1.2, c0/lambda_L/1e9*1.2)
    # plt.xlim(-c0 / lambda_L / 1e9 * 1.1, c0 / lambda_L / 1e9 * 1.1)
    # # plt.xlim(0.5, c0/lambda_0/1e9*1.5)
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("Impedance")
    # plt.title("Dipole Impedance")
    # plt.tight_layout()
    # plt.show()
    fig, ax = plt.subplots(figsize = (6, 4))


    ax.plot(np.fft.fftshift(freqs_hz) / 1e9, np.fft.fftshift(Z_dipole.real), label = "real",
            color = 'b')
    ax.plot(np.fft.fftshift(freqs_hz) / 1e9, np.fft.fftshift(Z_dipole.imag), label = "imageinary",
            color = 'r')

    # ax.vlines(c0.get() / lambda_0 / 1e9, -200, 500, colors = "k")
    # ax.vlines(c0.get() / lambda_L / 1e9, -200, 500, colors = "k", linestyle = "dotted")
    # ax.vlines(c0.get() / lambda_U / 1e9, -200, 500, colors = "k", linestyle = "dotted")
    resonant_freq = c0.get() / lambda_0 / 1e9

    x_freq = np.fft.fftshift(freqs_hz) / 1e9
    y_real = np.fft.fftshift(Z_dipole.real)
    y_imag = np.fft.fftshift(Z_dipole.imag)

    idx = np.abs(x_freq - resonant_freq).argmin()


    real_intersect = y_real[idx]
    imag_intersect = y_imag[idx]


    ax.vlines(resonant_freq, -200, 500, colors = "k")
    ax.vlines(c0.get() / lambda_L / 1e9, -200, 500, colors = "k", linestyle = "dotted")
    ax.vlines(c0.get() / lambda_U / 1e9, -200, 500, colors = "k", linestyle = "dotted")

    ax.plot(resonant_freq, real_intersect, 'bo', markersize = 6)
    ax.plot(resonant_freq, imag_intersect, 'ro', markersize = 6)

    ax.annotate(f'({resonant_freq:.2f}, {real_intersect:.2f})',
                xy = (resonant_freq, real_intersect),
                xytext = (10, 10),
                textcoords = 'offset points',
                fontsize = 8,
                arrowprops = dict(arrowstyle = '->', color = 'blue'))

    ax.annotate(f'({resonant_freq:.2f}, {imag_intersect:.2f})',
                xy = (resonant_freq, imag_intersect),
                xytext = (10, -20),
                textcoords = 'offset points',
                fontsize = 8,
                arrowprops = dict(arrowstyle = '->', color = 'red'))


    ax.set_ylim(-200, 300)
    ax.set_xlim(0.5, 1.6)


    ax.legend()
    ax.grid(True)
    # ax.set_xlim(-c0 / lambda_L / 1e9 * 1.1, c0 / lambda_L / 1e9 * 1.1)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Impedance(Z0)")
    ax.set_title("Dipole Impedance")

    plt.tight_layout()
    plt.show()
    target_freq = float(c0 / lambda_0)
    index = np.argmin(np.abs(freqs_hz - target_freq))
    print(f'target freq {target_freq}')
    print(f'impedence at resonant Ghz is {Z_dipole[index]}')

    np.savetxt(data_save_path, Z_dipole)




def plot_final_fields(Ex, Ey, Ez, Hx, Hy, Hz, Nx, Ny, Nz):
    kx = Nx // 2
    ky = Ny // 2
    kz = Nz // 2
    plt.figure(figsize = (15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(Ex[kx, :, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Ex at z = center')
    plt.xlabel("y")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(Ey[:, ky, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Ey at z = center')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(Ez[:, :, kz], cmap = 'RdBu', origin = 'lower')
    plt.title('Ez at z = center')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(Hx[kx, :, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Hx at z = center')
    plt.xlabel("y")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(Hy[:, ky, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Hy at z = center')
    plt.xlabel("x")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(Hz[:, :, kz], cmap = 'RdBu', origin = 'lower')
    plt.title('Hz at z = center')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar()

    plt.suptitle('Final field Distribution (z = center slice)')
    plt.tight_layout()
    plt.show()


def plot_probe_fields(t, Ex_record, Ey_record, Ez_record, Hx_record, Hy_record, Hz_record):
    plt.figure(figsize = (10, 6))
    plt.plot(t, Ex_record, label = 'Ex', linewidth = 1.5)
    plt.plot(t, Ey_record, label = 'Ey', linewidth = 3, linestyle = 'dotted')
    plt.plot(t, Ez_record, label = 'Ez', linewidth = 1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Electric Field at PML Inner Layer')
    plt.title('Electric Field vs Time at Probe Point')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize = (10, 6))
    plt.plot(t, Hx_record, label = 'Hx', linewidth = 1.5)
    plt.plot(t, Hy_record, label = 'Hy', linewidth = 3, linestyle = 'dotted')
    plt.plot(t, Hz_record, label = 'Hz', linewidth = 1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnetic Field at PML Inner Layer')
    plt.title('Magnetic Field vs Time at Probe Point')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_quadrants(ax, array, fixed_coord, cmap):
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    min_val = array.min() / 8
    max_val = array.max() / 8
    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:ny // 2, 0:nz // 2]
            X = nx // 2 * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride = 1, cstride = 1,
                            facecolors = facecolors, shade = False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:nx // 2, 0:nz // 2]
            Y = ny // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride = 1, cstride = 1,
                            facecolors = facecolors, shade = False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:nx // 2, 0:ny // 2]
            Z = nz // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride = 1, cstride = 1,
                            facecolors = facecolors, shade = False)


def figure_3D_array_slices(x, y, z, cmap = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_box_aspect(
        (max(x.shape[0], y.shape[0], z.shape[0]), max(x.shape[1], y.shape[1], z.shape[1]),
         max(x.shape[2], y.shape[2], z.shape[2])))
    plot_quadrants(ax, x, 'x', cmap = cmap)
    plot_quadrants(ax, y, 'y', cmap = cmap)
    plot_quadrants(ax, z, 'z', cmap = cmap)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)", labelpad = 20)
    return fig, ax

if __name__ == "__main__":
    main()