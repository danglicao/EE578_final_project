import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

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

    # Spatial Grid Definition
    x_min, x_max = -1500, 1500
    y_min, y_max = -1500, 1500

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1



    nt = int(1e6)
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
    pml_thickness = 15
    # sigma_max = 1e10
    sigma_max = -(np.float32(3 + 1) / np.float32(4)) * (c0 / np.float32(pml_thickness)) * np.log(
        np.float32(1e-10))

    # sigma_x, sigma_y = pml_profile(sigma_max, pml_thickness, Nx, Ny)
    sigma_x_vec, sigma_y_vec = pml_profile(sigma_max, pml_thickness, Nx, Ny)
    sigma_x_2d, sigma_y_2d = np.meshgrid(sigma_x_vec, sigma_y_vec,
                                                     indexing = 'ij')

    Ez_record = np.zeros(int(nt), dtype = np.float32)

    print(Bx[1:-1, 1:-1].shape)
    print(sigma_y_2d.shape)
    print((0.5 * (sigma_y_2d[2:-1, 1:-1] + sigma_y_2d[1:-2, 1:-1])).shape)
    print(Hx[1:-1, 1:-1].shape)
    print(Hy[1:-1, 1:-1].shape)






    # main loop
    for n in tqdm(range(nt)):
        #add source
        Ez[i_x_src][i_y_src] += gaussian_source(n, dt, sigma, omega_0)

        Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old = update_equations(Dz, Ez, Hx, Hy, Bx, By,
            Dz_old, Bx_old, By_old,
            sigma_x_2d, sigma_y_2d, epsilon, mu,
            dt, dx, dy)
        Ez_record[n] = Ez[i_x_prob][i_y_prob]

    #plotting
    fig, ax = plt.subplots()
    ax.imshow(Ez, extent = [x_min,x_max,y_min,y_max])
    square = patches.Rectangle((x_min+pml_thickness*dx, y_min+pml_thickness*dy), x_max-x_min-2*dx*pml_thickness, y_max-y_min-2*dy*pml_thickness,fc='none', ec='r')
    ax.add_patch(square)
    #print(Ez)
    plt.show()

    plt.plot(np.arange(nt), Ez_record)
    plt.xlabel('Time (s)')
    plt.ylabel('Ez at probe point')
    plt.title('Field at Probe Point')
    plt.grid(True)
    plt.show()

def gaussian_source(n, dt, sigma, omega0):
    t_now = (n - 0.5) * dt
    t0 = 4 * sigma
    return np.exp(-((t_now - t0) / sigma)**2) * np.sin(omega0 * (t_now - t0))

def sigma_profile(sigma_max, pml_thickness, distance):
    poly_deg = 3
    return sigma_max * (distance/pml_thickness)**poly_deg

# def pml_profile(sigma_max, pml_thickness, Nx, Ny):
#     sigma_x = np.zeros((Nx, Ny))
#     sigma_y = np.zeros((Nx, Ny))
#     for i in range(pml_thickness):
#         sigma_x[i, :] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
#         sigma_x[-1 - i, :] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
#     for j in range(pml_thickness):
#         sigma_y[:, j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
#         sigma_y[:, -1 - j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
#     return sigma_x, sigma_y

def pml_profile(sigma_max, pml_thickness, Nx, Ny):
    sigma_x = np.zeros(Nx)
    sigma_y = np.zeros(Ny)

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
    # Dz, Ez
    Dz[1:-1, 1:-1] = ((1 - 0.5 * dt * sigma_y[1:-1, 1:-1]) / (1 + 0.5 * dt * sigma_y[1:-1, 1:-1])) * Dz[1:-1, 1:-1] + \
        dt / (1 + 0.5 * dt * sigma_y[1:-1, 1:-1]) / epsilon[1:-1, 1:-1] * \
        ((Hy[1:, 1:-1] - Hy[:-1,1:-1]) / dx - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy)
    Ez[1:-1, 1:-1] = ((1 - 0.5 * dt  * sigma_x[1:-1, 1:-1]) / (1 + 0.5 * dt * sigma_x[1:-1, 1:-1])) * Ez[1:-1, 1:-1] + \
        1 /  (1 + 0.5 * dt  * sigma_x[1:-1, 1:-1]) * (Dz[1:-1, 1:-1] - Dz_old[1:-1, 1:-1])

    # Bx, Hx
    sigma_y_Bx = 0.5 * (sigma_y[1:-1, 2:-1] + sigma_y[1:-1, 1:-2])
    sigma_x_Hx = 0.5 * (sigma_x[1:-1, 2:-1] + sigma_x[1:-1, 1:-2])
    mu_Hx = 0.5 * (mu[:, :-1] + mu[:, 1:])
    Bx[1:-1, 1:-1] = ((1 - dt/2 * sigma_y_Bx) / (1 + dt/2 * sigma_y_Bx)) * Bx[1:-1, 1:-1] - \
        dt / (1 + dt/2 * sigma_y_Bx) * 1 * (Ez[1:-1, 2:-1] - Ez[1:-1, 1:-2])/dy
    # Hx[1:-1, 1:-1] = Hx[1:-1, 1:-1] + \
    #         (1 + dt / 2 * sigma_x_Hx) * Bx[1:-1,1:-1] - \
    #         (1 - dt / 2 * sigma_x_Hx) * Bx_old[1:-1, 1:-1]
    Hx[1:-1, 1:-1] = Hx[1:-1, 1:-1] + \
            (Bx[1:-1,1:-1]-Bx_old[1:-1, 1:-1]) +\
            0.5 * dt * sigma_x_Hx *(Bx[1:-1,1:-1]+Bx_old[1:-1, 1:-1])

        # By, Hy
    sigma_x_By = 0.5 * (sigma_x[2:-1, 1:-1] + sigma_x[1:-2, 1:-1])
    sigma_y_Hy = 0.5 * (sigma_y[2:-1, 1:-1] + sigma_y[1:-2, 1:-1])
    mu_Hy = 0.5 * (mu[:-1, :] + mu[1:, :])
    By[1:-1, 1:-1] = ((1 - dt/2 * sigma_x_By) / (1 + dt/2 * sigma_x_By)) * By[1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x_By) / 1 * (Ez[2:-1, 1:-1] - Ez[1:-2, 1:-1]) / dx
    # Hy[1:-1, 1:-1] = Hy[1:-1, 1:-1] + (1 + dt / 2 * sigma_y_Hy) * By[1:-1, 1:-1] - \
    #     (1 - dt / 2 * sigma_y_Hy) * By_old[1:-1, 1:-1]
    Hy[1:-1, 1:-1] = Hy[1:-1, 1:-1] + \
                     (By[1:-1, 1:-1] - By_old[1:-1, 1:-1]) + \
                     0.5 * dt * sigma_y_Hy * (By[1:-1, 1:-1] + By_old[1:-1, 1:-1])

    Dz_old = np.copy(Dz)
    Bx_old = np.copy(Bx)
    By_old = np.copy(By)

    return Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old

if __name__ == "__main__":
    main()