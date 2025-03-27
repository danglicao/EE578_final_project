import numpy as cp
import matplotlib.pyplot as plt
import cupy as cp

def main():
    # Simulation parameters
    epsilon0 = 1.0
    mu0 = 1.0
    c0 = 1.0
    lambda_0 = 950
    lambda_U = 1000
    lambda_L = 900
    dx = dy = dz = 20
    dt = dx / (c0 * cp.sqrt(3))

    x_min, x_max = -1500, 1500
    y_min, y_max = -1500, 1500
    z_min, z_max = -1500, 1500

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = int(50)
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
    pml_thickness = 20
    sigma_max = (3 + 1) * epsilon0 * c0 / (2 * dx)
    sigma_x_vec, sigma_y_vec, sigma_z_vec = pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz)
    # sigma_x_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    # sigma_y_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    # sigma_z_3d = cp.zeros((Nx, Ny, Nz), dtype = cp.float32)
    sigma_x_3d, sigma_y_3d, sigma_z_3d = cp.meshgrid(sigma_x_vec, sigma_y_vec, sigma_z_vec,
                                                     indexing = 'ij')

    #main loop
    for n in range(nt):
        #add source
        Ex[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)

        Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old =update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                         Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                         sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu,
                         dt, dx, dy, dz)

        Ex_record[n] = Ex[i_x_prob, i_y_prob, i_z_prob]

    #plotting
    t = cp.arange(nt) * dt
    plt.plot(t, Ex_record)
    plt.xlabel('Time (s)')
    plt.ylabel('Ex at probe point')
    plt.title('Field at Probe Point')
    plt.grid(True)
    plt.show()

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
            (Hz[1:-1, 2:-1, 1:-1] - Hz[1:-1, 1:-2, 1:-1]) / (2 * dy) -
            (Hy[1:-1, 1:-1, 2:-1] - Hy[1:-1, 1:-1, 1:-2]) / (2 * dz)
        )


    sigma_z_Dy = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
    sigma_z_Dy = sigma_z_Dy[1:-1, :, 1:-1]
    Dy[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_z_Dy) / (1 + dt/2 * sigma_z_Dy)) * Dy[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_z_Dy) * (
            (Hx[1:-1, 1:-1, 1:-2] - Hx[1:-1, 1:-1, 2:-1]) / dz -
            (Hz[2:-1, 1:-1, 1:-1] - Hz[1:-2, 1:-1, 1:-1]) / dx
        )

    sigma_x_Dz = 0.5 * (sigma_x[:, :, :-1] + sigma_x[:, :, 1:])
    sigma_x_Dz = sigma_x_Dz[1:-1, 1:-1, :]
    Dz[1:-1, 1:-1, 1:-1] = ((1 - dt/2 * sigma_x_Dz) / (1 + dt/2 * sigma_x_Dz)) * Dz[1:-1, 1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x_Dz) * (
            (Hy[1:-2, 1:-1, 1:-1] - Hy[2:-1, 1:-1, 1:-1]) / dx -
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
    # 内部区域在 j 方向取 [1:-1]
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








if __name__ == "__main__":
    main()