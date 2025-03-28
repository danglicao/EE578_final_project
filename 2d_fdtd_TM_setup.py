import numpy as np
import matplotlib.pyplot as plt

def main():
    # Simulation parameters
    epsilon0 = 1.0
    mu0 = 1.0
    c0 = 1.0
    lambda_0 = 600
    lambda_U = 1200
    lambda_L = 900
    dx = dy = 20
    dt = dx / (c0 * np.sqrt(2))
    poly_deg = 3

    # Spatial Grid Definition
    x_min, x_max = -1500, 1500
    y_min, y_max = -1500, 1500

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1

    nt = 230
    x_src, y_src= 0, 0
    x_prob, y_prob = 1000, 0
    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))

    omega_0 = 2 * np.pi * c0 / lambda_0
    sigma = (2 / omega_0) * (lambda_0 / (lambda_U - lambda_L))

    Ez = np.zeros((Nx, Ny), dtype = np.float32)
    Dz = np.zeros((Nx, Ny), dtype = np.float32)
    Dz_old = np.zeros((Nx, Ny), dtype = np.float32)
    
    Hx = np.zeros((Nx, Ny-1), dtype = np.float32)
    Bx = np.zeros((Nx, Ny-1), dtype = np.float32)
    Bx_old = np.zeros((Nx, Ny-1), dtype = np.float32)
    
    Hy = np.zeros((Nx-1, Ny), dtype = np.float32)
    By = np.zeros((Nx-1, Ny), dtype = np.float32)
    By_old = np.zeros((Nx-1, Ny), dtype = np.float32)
    
    epsilon = np.ones((Nx, Ny), dtype = np.float32) * epsilon0
    mu = np.ones((Nx, Ny), dtype = np.float32) * mu0

    #PML parameters
    pml_thickness = 20
    sigma_max = 1e10
    sigma_x_vec, sigma_y_vec = pml_profile(sigma_max, pml_thickness, Nx, Ny)

    #main loop
    for n in range(nt):
        #add source
        Ex[i_x_src][i_y_src][i_z_src] += gaussian_source(n, dt, sigma, omega_0)

        Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old = update_equations(Dz, Ez, Hx, Hy, Bx, By,
            Dz_old, Bx_old, By_old,
            sigma_x, sigma_y, epsilon, mu,
            dt, dx, dy)

        Ex_record[n] = Ex[i_x_prob, i_y_prob]

    plot_final_fields(Ex, Ey, Nx, Ny)
    #plotting
    t = np.arange(nt) * dt
    plt.plot(t, Ex_record)
    plt.xlabel('Time (s)')
    plt.ylabel('Ex at probe point')
    plt.title('Field at Probe Point')
    plt.grid(True)
    plt.show()

def gaussian_source(n, dt, sigma, omega0):
    t_now = (n - 0.5) * dt
    t0 = 4 * sigma
    return np.exp(-((t_now - t0) / sigma)**2) * np.sin(omega0 * (t_now - t0))

def sigma_profile(sigma_max, pml_thickness, distance):
    return sigma_max * (distance/pml_thickness)**poly_deg

def pml_profile(sigma_max, pml_thickness, Nx, Ny):
    sigma_x = np.zeros(Nx)
    sigma_y = np.zeros(Ny)
    sigma_z = np.zeros(Nz)
    for i in range(pml_thickness):
        sigma_x[i] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
        sigma_x[-1 - i] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
    for j in range(pml_thickness):
        sigma_y[j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
        sigma_y[-1 - j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
    return sigma_x, sigma_y

def update_equations(Dz, Ez, Hx, Hy, Bx, By,
                     Dz_old, Bx_old, By_old,
                     sigma_x, sigma_y, epsilon, mu,
                     dt, dx, dy):

    Dz[1:-1, 1:-1] = ((1 - dt/2 * sigma_x[1:-1]) / (1 + dt/2 * sigma_x[1:-1])) * Dz[1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x[1:-1]) / epsilon[1:-1] * (
            (Hy[1:-2, 1:-1] - Hy[2:-1, 1:-1]) / dx -
            (Hx[1:-1, 2:-1] - Hx[1:-1, 1:-2]) / dy
        )

    Ez[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y) / (1 + dt / 2 * sigma_y)) * Ez[1:-1, 1:-1, 1:-1] + \
                           1 /  (1 + dt / 2 * sigma_y) * \
                           (Dz[1:-1, 1:-1, 1:-1] - Dz_old[1:-1, 1:-1, 1:-1])

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

    Dz_old = np.copy(Dz)
    Bx_old = np.copy(Bx)
    By_old = np.copy(By)

    return Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old

if __name__ == "__main__":
    main()