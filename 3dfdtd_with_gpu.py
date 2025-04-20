import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm


def main():
    # Simulation parameters

    epsilon0 = cp.float32(1.0)
    mu0 = cp.float32(1.0)
    c0 = cp.float32(1.0)


    lambda_0 = cp.float32(1000)
    lambda_U = cp.float32(1500)
    lambda_L = cp.float32(500)

    dx = dy = dz = cp.float32(20)


    dt = 0.99*dx / (c0 * cp.sqrt(cp.float32(3)))


    x_min, x_max = -1500, 1500
    y_min, y_max = -1500, 1500
    z_min, z_max = -1500, 1500


    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = int(3e3)


    x_src, y_src, z_src = 0, 0, 0
    x_prob, y_prob, z_prob = 1000, 0, 0


    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))
    i_z_src = int(round((z_src - z_min) / dz))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_z_prob = int(round((z_prob - z_min) / dz))
    Ex_record = cp.zeros(nt, dtype = cp.float32)
    Ey_record = cp.zeros(nt, dtype = cp.float32)
    Ez_record = cp.zeros(nt, dtype = cp.float32)

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


    #PML parameters
    pml_thickness = int(12)
    power_reflection_coefficient = cp.float32(1e-10)
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
    #main loop
    for n in tqdm(range(nt)):
        #add source
        # Ex[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)
        # Ey[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)
        Ez[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)

        Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old =update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                         Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                         sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu,
                         dt, dx, dy, dz)

        Ex_record[n] = Ex[i_x_prob, i_y_prob, i_z_prob]
        Ey_record[n] = Ey[i_x_prob, i_y_prob, i_z_prob]
        Ez_record[n] = Ez[i_x_prob, i_y_prob, i_z_prob]
        if n == 230:
            Ex_time_record = cp.copy(Ex)
            Ey_time_record = cp.copy(Ey)
            Ez_time_record = cp.copy(Ez)


    Ex_time_record_cpu = Ex_time_record.get()
    Ey_time_record_cpu = Ey_time_record.get()
    Ez_time_record_cpu = Ez_time_record.get()

    gpu_vars = {
        'Dx': Dx, 'Dy': Dy, 'Dz': Dz,
        'Ex': Ex, 'Ey': Ey, 'Ez': Ez,
        'Hx': Hx, 'Hy': Hy, 'Hz': Hz,
        'Bx': Bx, 'By': By, 'Bz': Bz,
        'Dx_old': Dx_old, 'Dy_old': Dy_old, 'Dz_old': Dz_old,
        'Bx_old': Bx_old, 'By_old': By_old, 'Bz_old': Bz_old,
    }

    cpu_vars = {name + '_cpu': arr.get() for name, arr in gpu_vars.items()}
    Ex_record_cpu = Ex_record.get()
    Ey_record_cpu = Ey_record.get()
    Ez_record_cpu = Ez_record.get()
    #plotting

    #plot_final_fields(cpu_vars['Ex_cpu'], cpu_vars['Ey_cpu'], cpu_vars['Ez_cpu'], Nx, Ny, Nz)
    plot_final_fields(Ex_time_record_cpu, Ey_time_record_cpu, Ez_time_record_cpu,Nx, Ny, Nz)
    t = np.arange(nt) * dt.get()
    #print(Ex_record_cpu.shape)
    #print(Ex_record_cpu)
    plot_probe_fields(t, Ex_record_cpu, Ey_record_cpu, Ez_record_cpu)

def gaussian_source(n, dt, sigma, omega0):
    t_now = (n - 0.5) * dt
    t0 = 4 * sigma
    return cp.exp(-((t_now - t0) / sigma)**2) * cp.sin(omega0 * (t_now - t0))

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
            (Hz[1:-1, 2:-1, 1:-1] - Hz[1:-1, 1:-2, 1:-1]) / (dy) -
            (Hy[1:-1, 1:-1, 2:-1] - Hy[1:-1, 1:-1, 1:-2]) / (dz)
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
    plt.imshow(Ex[kx, :, :], cmap='RdBu', origin='lower')
    plt.title('Ex at x = center')
    plt.colorbar()

    # Ey 平面图：取 Ey[:, :, kz]
    plt.subplot(1, 3, 2)
    plt.imshow(Ey[ky, :, :], cmap='RdBu', origin='lower')
    plt.title('Ey at y = center')
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




if __name__ == "__main__":
    main()