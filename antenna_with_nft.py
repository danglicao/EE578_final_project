import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm
import matplotlib.animation as animation
import pyvista as pv
#all wavelength/boundary length in mm




def main():
    # Simulation parameters

    epsilon0 = cp.float32(1.0)
    mu0 = cp.float32(1.0)
    c0 = cp.float32(1.0)
    epsilonR = cp.float32(12.0)


    lambda_0 = cp.float32(125) #mm
    lambda_U = cp.float32(132) #mm
    lambda_L = cp.float32(118.5) #mm

    dx = dy = dz = cp.float32(2.5)  # mm, spatial step size


    dt = 0.99*dx / (c0 * cp.sqrt(cp.float32(3)))



    x_min, x_max = -200, 200
    y_min, y_max = -200, 200
    z_min, z_max = -200, 200


    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = int(3e3)
    record_time = 108

    #Antenna hyper parameters
    L = cp.float32(lambda_0 / (2))
    L_rel = L//dz
    ic, jc, kc = Nx // 2, Ny // 2, Nz // 2

    half_len = L / 2
    half_len_rel = int(L_rel // 2)

    i_z_dipole_start = kc - half_len_rel
    i_z_dipole_end = kc + half_len_rel + 1

    x_src, y_src, z_src = 0, 0, 0
    x_prob, y_prob, z_prob = 180, 0, 0

    # x_epsilon_upper, y_epsilon_upper, z_epsilon_upper = -90, -90, -90
    # x_epsilon_lower, y_epsilon_lower, z_epsilon_lower = 90, 90, 90


    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))
    i_z_src = int(round((z_src - z_min) / dz))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_z_prob = int(round((z_prob - z_min) / dz))

    # i_x_epsilon_upper = int(round((x_epsilon_upper - x_min) / dx))
    # i_y_epsilon_upper = int(round((y_epsilon_upper - y_min) / dy))
    # i_z_epsilon_upper = int(round((z_epsilon_upper - z_min) / dz))
    # i_x_epsilon_lower = int(round((x_epsilon_lower - x_min) / dx))
    # i_y_epsilon_lower = int(round((y_epsilon_lower - y_min) / dy))
    # i_z_epsilon_lower = int(round((z_epsilon_lower - z_min) / dz))



    Ex_record = cp.zeros(nt, dtype = cp.float32)
    Ey_record = cp.zeros(nt, dtype = cp.float32)
    Ez_record = cp.zeros(nt, dtype = cp.float32)

    Ez_gap_record_left = cp.zeros(nt, dtype = cp.float32)
    Ez_gap_record_right = cp.zeros(nt, dtype = cp.float32)
    j_record = cp.zeros(nt, dtype = cp.float32)


    omega_0 = 2 * cp.pi * c0 / lambda_0
    sigma = (2 / omega_0) * (lambda_0 / (lambda_U - lambda_L))

    Ex = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float32)
    Ey = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float32)
    Ez = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float32)
    Hx = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float32)
    Hy = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float32)
    Hz = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float32)
    Bx = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float32)
    By = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float32)
    Bz = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float32)
    Dx = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float32)
    Dy = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float32)
    Dz = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float32)
    Bx_old = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float32)
    By_old = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float32)
    Bz_old = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float32)
    Dx_old = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float32)
    Dy_old = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float32)
    Dz_old = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float32)
    epsilon = cp.ones((Nx, Ny, Nz), dtype = cp.float32) * epsilon0
    mu = cp.ones((Nx, Ny, Nz), dtype = cp.float32) * mu0
    # epsilon[:,i_y_epsilon_upper:i_y_epsilon_lower, :] = epsilon0 * epsilonR


    #PML parameters
    pml_thickness = int(8)
    power_reflection_coefficient = cp.float32(1e-8)
    # sigma_max = cp.float32(4) * epsilon0 * c0 / (2 * dx)
    sigma_max = -(cp.float32(3+1) / cp.float32(4)) * (c0 / cp.float32(pml_thickness)) * cp.log(cp.float32(power_reflection_coefficient))

    sigma_x_vec, sigma_y_vec, sigma_z_vec = pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz)
    # sigma_x_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    # sigma_y_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    # sigma_z_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    sigma_x_3d, sigma_y_3d, sigma_z_3d = cp.meshgrid(sigma_x_vec, sigma_y_vec, sigma_z_vec,
                                                     indexing = 'ij')

    Ex_time_record = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float32)
    Ey_time_record = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float32)
    Ez_time_record = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float32)


    #用电导率模拟pec
    sigma_x_3d[ic-1, jc-1, i_z_dipole_start : i_z_dipole_end] = 1e8
    sigma_y_3d[ic-1, jc-1, i_z_dipole_start : i_z_dipole_end] = 1e8
    sigma_z_3d[ic-1, jc-1, i_z_dipole_start : i_z_dipole_end] = 1e8
    sigma_x_3d[ic + 1, jc + 1, i_z_dipole_start: i_z_dipole_end] = 1e8
    sigma_y_3d[ic + 1, jc + 1, i_z_dipole_start: i_z_dipole_end] = 1e8
    sigma_z_3d[ic + 1, jc + 1, i_z_dipole_start: i_z_dipole_end] = 1e8
    # sigma_x_3d[ic, jc, i_z_dipole_start: i_z_dipole_end] = 1e8
    # sigma_y_3d[ic, jc, i_z_dipole_start: i_z_dipole_end] = 1e8
    # sigma_z_3d[ic, jc, i_z_dipole_start: i_z_dipole_end] = 1e8
    #for source
    z_coords = cp.arange(i_z_dipole_start, i_z_dipole_end)
    z_rel = (z_coords - kc) * dz
    I_dipole = I_profile(lambda_0, c0, z_rel, L)
    # main loop
    for n in tqdm(range(nt)):
        # Ez[ic - 1, jc - 1, i_z_dipole_start:i_z_dipole_end] += 0.5 *  gaussian_source(n,
        #                                                                                         dt,
        #                                                                                         sigma,
        #                                                                                         omega_0)
        # Ez[ic + 1, jc + 1, i_z_dipole_start:i_z_dipole_end] += 0.5 * gaussian_source(n,
        #                                                                                         dt,
        #                                                                                         sigma,
        #                                                                                         omega_0)

        # Ez[ic - 1, jc - 1, i_z_dipole_start:i_z_dipole_end] += 0.5 * I_dipole * gaussian_source(n,
        #                                                                                     dt,
        #                                                                                     sigma,
        #                                                                                     omega_0)
        # Ez[ic + 1, jc + 1, i_z_dipole_start:i_z_dipole_end] += 0.5 * I_dipole * gaussian_source(n,
        #                                                                                     dt,
        #                                                                                     sigma,
        #                                                                                     omega_0)

        #add source
        # Ex[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)
        # Ey[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)
        # Ez[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)
        # Ez[ic, jc, i_z_dipole_start:i_z_dipole_end] += 0.5 * I_dipole * gaussian_source(n, dt, sigma, omega_0)
        #point source
        # Ez[ic, jc, kc] += gaussian_source(n,
        #                                                                dt,
        #                                                                sigma,
        #                                                                omega_0)
        #gaussian source
        Ez[ic, jc, i_z_dipole_start:i_z_dipole_end] += gaussian_source(n,
                                                                                                dt,
                                                                                                sigma,
                                                                                                omega_0)
        # gaussian source with dipole current profile
        # Ez[ic, jc, i_z_dipole_start:i_z_dipole_end] += I_dipole * gaussian_source(n,
        #                                                                                     dt,
        #                                                                                     sigma,
        #                                                                                     omega_0)
        #sine source
        # Ez[ic, jc, i_z_dipole_start:i_z_dipole_end] += sine_source(n,
        #                                                                                         dt,
        #                                                                                         sigma,
        #                                                                                         omega_0)
        # sine source with dipole current profile
        # Ez[ic, jc, i_z_dipole_start:i_z_dipole_end] += I_dipole * sine_source(n,
        #                                                                          dt,
        #                                                                          sigma,
        #                                                                          omega_0)


        Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old =update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                         Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                         sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu,
                         dt, dx, dy, dz)

        # Ex[ic,jc,i_z_dipole_start:i_z_dipole_end] = 0
        # Ey[ic,jc,i_z_dipole_start:i_z_dipole_end] = 0
        # Ez[ic,jc,i_z_dipole_start:i_z_dipole_end] = 0

        Ex_record[n] = Ex[i_x_prob, i_y_prob, i_z_prob]
        Ey_record[n] = Ey[i_x_prob, i_y_prob, i_z_prob]
        Ez_record[n] = Ez[i_x_prob, i_y_prob, i_z_prob]

        Ez_gap_record_left[n] = Ez[ic, jc, i_z_dipole_start+1]
        Ez_gap_record_right[n] = Ez[ic, jc, i_z_dipole_end - 1]
        j_record[n] = gaussian_source(n, dt,sigma,omega_0)



    Ex_time_record_cpu = Ex_time_record.get()
    Ey_time_record_cpu = Ey_time_record.get()
    Ez_time_record_cpu = Ez_time_record.get()




    Ex_record_cpu = Ex_record.get()
    Ey_record_cpu = Ey_record.get()
    Ez_record_cpu = Ez_record.get()

    t = np.arange(nt) * dt.get()
    plot_probe_fields(t, Ex_record_cpu, Ey_record_cpu, Ez_record_cpu)

    Ez_gap_record_left_cpu = Ez_gap_record_left.get()
    Ez_gap_record_right_cpu = Ez_gap_record_right.get()
    dEz_cpu = Ez_gap_record_right_cpu - Ez_gap_record_left_cpu
    j_record_cpu = j_record.get()
    plt.plot(dEz_cpu, label = "dEz")
    plt.plot(j_record_cpu, label = "J")
    plt.legend()
    plt.title("Time-domain Signals")

    #calculate impedance
    c_real = 3e8  # m/s
    dx_real = dy_real = dz_real = 2.5e-3 #m
    dt_real = 0.99 * dx_real / (c_real * np.sqrt(3)) # s

    Vz_f = np.fft.fft(dEz_cpu)
    J_f = np.fft.fft(j_record_cpu)

    V_f = Vz_f * dz_real
    I_f = J_f * dx_real * dy_real* L_rel

    Z_f = V_f / I_f
    print(Z_f)

    #plotting
    freqs_hz = np.fft.fftfreq(nt, d=dt_real)  # Hz
    freqs_ghz = freqs_hz * 1e-9

    half = nt // 2
    Z_f = Z_f[:half]
    freqs_ghz = freqs_ghz[:half]

    plt.figure(figsize = (10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(freqs_ghz, np.abs(Z_f))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|Z(f)| (Ohms)")
    plt.title("Input Impedance Magnitude")

    plt.subplot(1, 2, 2)
    plt.plot(freqs_ghz, Z_f.real, label = "Re")
    plt.plot(freqs_ghz, Z_f.imag, label = "Im")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Z(f) (Ohms)")
    plt.title("Input Impedance (Re/Im)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def gaussian_source(n, dt, sigma, omega0):
    t_now = (n - 0.5) * dt
    t0 = 4 * sigma
    return cp.exp(-((t_now - t0) / sigma)**2) * cp.sin(omega0 * (t_now - t0))

def sine_source(n, dt, sigma, omega0, amplitude=1.0):
    t_now = (n - 0.5) * dt
    t0 = 4*sigma
    return amplitude * cp.sin(omega0 * (t_now - t0))

def I_profile( lambda_0, c0, position,  L):
    f0 = c0 / lambda_0
    k0 = 2 * cp.pi * f0 / c0
    # z_coords = cp.arange(i_z_dipole_start, i_z_dipole_end)
    # z_rel = (z_coords - kc) * dz
    return cp.sin(k0 * (L/ 2 - cp.abs(position)))



def sigma_profile(sigma_max, pml_thickness, distance):
    return sigma_max * (distance / pml_thickness)**3

def pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz):
    sigma_x = cp.zeros(Nx)
    sigma_y = cp.zeros(Ny)
    sigma_z = cp.zeros(Nz)
    for i in range(pml_thickness):
        sigma_x[i] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
        sigma_x[-1 - i] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
    for j in range(pml_thickness):
        sigma_y[j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
        sigma_y[-1 - j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
    for k in range(pml_thickness):
        sigma_z[k] = sigma_profile(sigma_max, pml_thickness, pml_thickness - k)
        sigma_z[-1 - k] = sigma_profile(sigma_max, pml_thickness, pml_thickness - k)
    return sigma_x, sigma_y, sigma_z

def update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                     Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                     sigma_x, sigma_y, sigma_z, epsilon, mu,
                     dt, dx, dy, dz):



    sigma_y_Dx = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])

    sigma_y_Dx = sigma_y_Dx[:, 1:-1, 1:-1]

    Dx[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_y_Dx) / (1 + dt/2 * sigma_y_Dx)) * Dx[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_y_Dx) * (
            (Hz[1:-1, 2:-1, 1:-1] - Hz[1:-1, 1:-2, 1:-1]) / (2 * dy) -
            (Hy[1:-1, 1:-1, 2:-1] - Hy[1:-1, 1:-1, 1:-2]) / (2 * dz)
        )


    sigma_z_Dy = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
    sigma_z_Dy = sigma_z_Dy[1:-1, :, 1:-1]
    Dy[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_z_Dy) / (1 + dt/2 * sigma_z_Dy)) * Dy[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_z_Dy) * (
            (Hx[1:-1, 1:-1, 2:-1] - Hx[1:-1, 1:-1, 1:-2]) / dz -
            (Hz[2:-1, 1:-1, 1:-1] - Hz[1:-2, 1:-1, 1:-1]) / dx
        )

    sigma_x_Dz = 0.5 * (sigma_x[:, :, :-1] + sigma_x[:, :, 1:])
    sigma_x_Dz = sigma_x_Dz[1:-1, 1:-1, :]
    Dz[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_x_Dz) / (1 + dt/2 * sigma_x_Dz)) * Dz[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x_Dz) * (
            (Hy[2:-1, 1:-1, 1:-1] - Hy[1:-2, 1:-1, 1:-1]) / dx -
            (Hx[1:-1, 2:-1, 1:-1] - Hx[1:-1, 1:-2, 1:-1]) / dy
        )

    sigma_z_Ex = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    sigma_z_Ex = sigma_z_Ex[:, 1:-1, 1:-1]

    sigma_x_Ex = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    sigma_x_Ex = sigma_x_Ex[:, 1:-1, 1:-1]

    epsilon_Ex = 0.5 * (epsilon[:-1, :, :] + epsilon[1:, :, :])
    epsilon_Ex = epsilon_Ex[:, 1:-1, 1:-1]


    Ex[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_Ex) / (1 + dt / 2 * sigma_z_Ex)) * Ex[1:-1, 1:-1,
                                                                                     1:-1] + \
                           1 / (epsilon_Ex * (1 + dt / 2 * sigma_z_Ex)) * \
                           ((1 + dt / 2 * sigma_x_Ex) * Dx[1:-1, 1:-1, 1:-1] - (
                                       1 - dt / 2 * sigma_x_Ex) * Dx_old[1:-1, 1:-1, 1:-1])


    sigma_x_Ey = 0.5 * (sigma_x[:, :-1, :] + sigma_x[:, 1:, :])
    sigma_x_Ey = sigma_x_Ey[1:-1, :, 1:-1]

    sigma_y_Ey = 0.5 * (sigma_y[:, :-1, :] + sigma_y[:, 1:, :])
    sigma_y_Ey = sigma_y_Ey[1:-1, :, 1:-1]

    epsilon_Ey = 0.5 * (epsilon[:, :-1, :] + epsilon[:, 1:, :])
    epsilon_Ey = epsilon_Ey[1:-1, :, 1:-1]


    Ey[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Ey) / (1 + dt / 2 * sigma_x_Ey)) * Ey[1:-1, 1:-1,
                                                                                     1:-1] + \
                           1 / (epsilon_Ey * (1 + dt / 2 * sigma_x_Ey)) * \
                           ((1 + dt / 2 * sigma_y_Ey) * Dy[1:-1, 1:-1, 1:-1] - (
                                       1 - dt / 2 * sigma_y_Ey) * Dy_old[1:-1, 1:-1, 1:-1])


    sigma_y_Ez = 0.5 * (sigma_y[:, :, :-1] + sigma_y[:, :, 1:])
    sigma_y_Ez = sigma_y_Ez[1:-1, 1:-1, :]

    sigma_z_Ez = 0.5 * (sigma_z[:, :, :-1] + sigma_z[:, :, 1:])
    sigma_z_Ez = sigma_z_Ez[1:-1, 1:-1, :]

    epsilon_Ez = 0.5 * (epsilon[:, :, :-1] + epsilon[:, :, 1:])
    epsilon_Ez = epsilon_Ez[1:-1, 1:-1, :]


    Ez[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Ez) / (1 + dt / 2 * sigma_y_Ez)) * Ez[1:-1, 1:-1,
                                                                                     1:-1] + \
                           1 / (epsilon_Ez * (1 + dt / 2 * sigma_y_Ez)) * \
                           ((1 + dt / 2 * sigma_z_Ez) * Dz[1:-1, 1:-1, 1:-1] - (
                                       1 - dt / 2 * sigma_z_Ez) * Dz_old[1:-1, 1:-1, 1:-1])

    sigma_y_Bx = 0.5 * (sigma_y[:, :-1, :] + sigma_y[:, 1:, :])
    sigma_y_Bx = 0.5 * (sigma_y_Bx[:, :, :-1] + sigma_y_Bx[:, :, 1:])
    sigma_y_Bx = sigma_y_Bx[1:-1, :, :]

    Bx[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_y_Bx) / (1 + dt/2 * sigma_y_Bx)) * Bx[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_y_Bx) * (
            (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
            (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
        )


    sigma_z_By = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    sigma_z_By = 0.5 * (sigma_z_By[:, :, :-1] + sigma_z_By[:, :, 1:])
    sigma_z_By = sigma_z_By[:, 1:-1, :]

    By[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_z_By) / (1 + dt/2 * sigma_z_By)) * By[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_z_By) * (
            (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
            (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
        )


    sigma_x_Bz = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    sigma_x_Bz = 0.5 * (sigma_x_Bz[:, :-1, :] + sigma_x_Bz[:, 1:, :])
    sigma_x_Bz = sigma_x_Bz[:, :, 1:-1]
    Bz[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_x_Bz) / (1 + dt/2 * sigma_x_Bz)) * Bz[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x_Bz) * (
        (Ex[1:-1, 1:, 1:-1] - Ex[1:-1, :-1, 1:-1]) / dy -
        (Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]) / dx
        )

    sigma_z_Hx = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
    sigma_z_Hx = 0.5 * (
                sigma_z_Hx[:, :, :-1] + sigma_z_Hx[:, :, 1:])

    sigma_z_Hx_in = sigma_z_Hx[1:-1, :, :]


    sigma_x_Hx = 0.5 * (sigma_x[:, :-1, :] + sigma_x[:, 1:, :])
    sigma_x_Hx = 0.5 * (sigma_x_Hx[:, :, :-1] + sigma_x_Hx[:, :, 1:])
    sigma_x_Hx_in = sigma_x_Hx[1:-1, :, :]


    mu_Hx = 0.5 * (mu[:, :-1, :] + mu[:, 1:, :])
    mu_Hx = 0.5 * (mu_Hx[:, :, :-1] + mu_Hx[:, :, 1:])
    mu_Hx_in = mu_Hx[1:-1, :, :]


    Hx[1:-1, 1:-1, 1:-1] = (
            ((1 - dt / 2 * sigma_z_Hx_in) / (1 + dt / 2 * sigma_z_Hx_in)) * Hx[1:-1, 1:-1, 1:-1] +
            1 / (mu_Hx_in * (1 + dt / 2 * sigma_z_Hx_in)) *
            (
                    (1 + dt / 2 * sigma_x_Hx_in) * Bx[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_x_Hx_in) * Bx_old[1:-1, 1:-1, 1:-1]
            )
    )


    sigma_x_Hy = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    sigma_x_Hy = 0.5 * (sigma_x_Hy[:, :, :-1] + sigma_x_Hy[:, :, 1:])
    sigma_x_Hy_in = sigma_x_Hy[:, 1:-1, :]


    sigma_y_Hy = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])
    sigma_y_Hy = 0.5 * (sigma_y_Hy[:, :, :-1] + sigma_y_Hy[:, :, 1:])
    sigma_y_Hy_in = sigma_y_Hy[:, 1:-1, :]


    mu_Hy = 0.5 * (mu[:-1, :, :] + mu[1:, :, :])
    mu_Hy = 0.5 * (mu_Hy[:, :, :-1] + mu_Hy[:, :, 1:])
    mu_Hy_in = mu_Hy[:, 1:-1, :]
    Hy[1:-1, 1:-1, 1:-1] = (
            ((1 - dt / 2 * sigma_x_Hy_in) / (1 + dt / 2 * sigma_x_Hy_in)) * Hy[1:-1, 1:-1, 1:-1] +
            1 / (mu_Hy_in * (1 + dt / 2 * sigma_x_Hy_in)) *
            (
                    (1 + dt / 2 * sigma_y_Hy_in) * By[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_y_Hy_in) * By_old[1:-1, 1:-1, 1:-1]
            )
    )


    sigma_y_Hz = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])
    sigma_y_Hz = 0.5 * (sigma_y_Hz[:, :-1, :] + sigma_y_Hz[:, 1:, :])
    sigma_y_Hz_in = sigma_y_Hz[:, :, 1:-1]


    sigma_z_Hz = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    sigma_z_Hz = 0.5 * (sigma_z_Hz[:, :-1, :] + sigma_z_Hz[:, 1:, :])
    sigma_z_Hz_in = sigma_z_Hz[:, :, 1:-1]


    mu_Hz = 0.5 * (mu[:-1, :, :] + mu[1:, :, :])
    mu_Hz = 0.5 * (mu_Hz[:, :-1, :] + mu_Hz[:, 1:, :])
    mu_Hz_in = mu_Hz[:, :, 1:-1]

    Hz[1:-1, 1:-1, 1:-1] = (
            ((1 - dt / 2 * sigma_y_Hz_in) / (1 + dt / 2 * sigma_y_Hz_in)) * Hz[1:-1, 1:-1, 1:-1] +
            1 / (mu_Hz_in * (1 + dt / 2 * sigma_y_Hz_in)) *
            (
                    (1 + dt / 2 * sigma_z_Hz_in) * Bz[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_z_Hz_in) * Bz_old[1:-1, 1:-1, 1:-1]
            )
    )

    Dx_old = cp.copy(Dx)
    Dy_old = cp.copy(Dy)
    Dz_old = cp.copy(Dz)
    Bx_old = cp.copy(Bx)
    By_old = cp.copy(By)
    Bz_old = cp.copy(Bz)

    return Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old


def plot_final_fields(Ex, Ey, Ez, Nx, Ny, Nz):
    kx = Nx // 2
    ky = Ny // 2
    kz = Nz // 2
    plt.figure(figsize=(15, 4))

    # Ex 平面图：取 Ex[:, :, kz]
    plt.subplot(1, 3, 1)
    plt.imshow(Ex[:, :, kz], cmap='RdBu', origin='lower')
    plt.title('Ex at z = center')
    plt.colorbar()

    # Ey 平面图：取 Ey[:, :, kz]
    plt.subplot(1, 3, 2)
    plt.imshow(Ey[:, :, kz], cmap='RdBu', origin='lower')
    plt.title('Ey at z = center')
    plt.colorbar()

    # Ez 平面图：取 Ez[:, :, kz]
    plt.subplot(1, 3, 3)
    plt.imshow(Ez[:, :, kz], cmap='RdBu', origin='lower')
    plt.title('Ez at z = center')
    plt.colorbar()

    plt.suptitle('Final E-field Distribution (z = center slice)')
    plt.tight_layout()
    plt.show()

def plot_probe_fields(t, Ex_record, Ey_record, Ez_record):

    plt.figure(figsize=(10, 6))
    plt.plot(t, Ex_record, label='Ex', linewidth=1.5)
    plt.plot(t, Ey_record, label='Ey', linewidth=1.5)
    plt.plot(t, Ez_record, label='Ez', linewidth=1.5)

    plt.xlabel('Time (s)')
    plt.ylabel('Electric Field at Probe Point')
    plt.title('Electric Field vs Time at Probe Point')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_final_fields_normalized(Ex, Ey, Ez, Nx, Ny, Nz):
    kx = Nx // 2
    ky = Ny // 2
    kz = Nz // 2

    # 取中心切片
    Ex_slice = Ex[:, :, kz]
    Ey_slice = Ey[:, :, kz]
    Ez_slice = Ez[:, :, kz]

    # 计算统一的 vmin 和 vmax
    abs_max = max(
        np.max(np.abs(Ex_slice)),
        np.max(np.abs(Ey_slice)),
        np.max(np.abs(Ez_slice))
    )
    vmin, vmax = -abs_max, abs_max

    plt.figure(figsize=(15, 4))

    # Ex 平面图
    plt.subplot(1, 3, 1)
    plt.imshow(Ex_slice.T, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Ex at z = center')
    plt.colorbar()

    # Ey 平面图
    plt.subplot(1, 3, 2)
    plt.imshow(Ey_slice.T, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Ey at z = center')
    plt.colorbar()

    # Ez 平面图
    plt.subplot(1, 3, 3)
    plt.imshow(Ez_slice.T, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Ez at z = center')
    plt.colorbar()

    plt.suptitle('Final E-field Distribution (z = center slice)')
    plt.tight_layout()
    plt.show()


def generate_field_animation(
        field_frames,
        save_path = "figs/field_animation.mp4",
        save_every = 1,
        cmap = "RdBu",
        vmin = None,
        vmax = None,
        fps = 30,
        dpi = 200
):

    n_frames = field_frames.shape[0]

    if vmin is None:
        vmin = field_frames.min()
    if vmax is None:
        vmax = field_frames.max()

    fig, ax = plt.subplots()
    im = ax.imshow(
        field_frames[0],  # Start with the first frame
        cmap = cmap,
        vmin = vmin,
        vmax = vmax,
        origin = "lower",
        aspect = "auto"
    )
    cb = plt.colorbar(im, ax = ax)
    cb.set_label("Field amplitude")

    def update(i):
        im.set_array(field_frames[i])
        ax.set_title(f"Time step {i * save_every}")
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames = n_frames, interval = 1000 / fps
    )
    ani.save(save_path, dpi = dpi, extra_args=['-vcodec', 'h264_nvenc'])
    print(f"✅ 动画保存成功：{save_path}")


def plot_electric_field_magnitude_3d(Ex, Ey, Ez, spacing = (1.0, 1.0, 1.0)):

    # 插值 Yee 网格分量到中心
    Ex_interp = 0.5 * (Ex[:-1, :, :] + Ex[1:, :, :])
    Ey_interp = 0.5 * (Ey[:, :-1, :] + Ey[:, 1:, :])
    Ez_interp = 0.5 * (Ez[:, :, :-1] + Ez[:, :, 1:])

    # 电场强度模长
    E_magnitude = np.sqrt(Ex_interp ** 2 + Ey_interp ** 2 + Ez_interp ** 2)

    # 构造 pyvista ImageData 网格
    Nx, Ny, Nz = E_magnitude.shape
    grid = pv.ImageData()
    grid.dimensions = np.array([Nx, Ny, Nz]) + 1
    grid.origin = (0.0, 0.0, 0.0)
    grid.spacing = spacing
    grid.cell_data["E_magnitude"] = E_magnitude.flatten(order = "F")

    # 初始化绘图器
    plotter = pv.Plotter()

    # 添加体积云图
    plotter.add_volume(
        grid, scalars = "E_magnitude",
        cmap = "plasma",
        opacity = [0.0, 0.05, 0.1, 0.3, 0.6, 1.0],  # 调整透明度以显示更多细节
        shade = True
    )

    # 添加 z = 中心 截面
    z_center = grid.bounds[5] / 2  # z_max / 2
    slice_z = grid.slice(normal = 'z', origin = (0, 0, z_center))
    plotter.add_mesh(slice_z, cmap = "coolwarm", opacity = 1.0, show_scalar_bar = True)

    # 显示图像
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    main()