import numpy as np
import matplotlib.pyplot as plt

def main():
    # Simulation parameters
    epsilon0 = 1.0
    mu0 = 1.0
    c0 = 1.0
    lambda_0 = 950
    lambda_U = 1000
    lambda_L = 900
    dx = dy = dz = 20
    dt = dx / (c0 * np.sqrt(3))

    x_min, x_max = -1500, 1500
    y_min, y_max = -1500, 1500
    z_min, z_max = -1500, 1500

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = int(1e5)
    x_src, y_src, z_src = 0, 0, 0
    x_prob, y_prob, z_prob = 1000, 0, 0
    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))
    i_z_src = int(round((z_src - z_min) / dz))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_z_prob = int(round((z_prob - z_min) / dz))

    omega_0 = 2 * np.pi * c0 / lambda_0
    sigma = (2 / omega_0) * (lambda_0 / (lambda_U - lambda_L))

    Ex = np.zeros((Nx + 1, Ny, Nz), dtype = np.float32)
    Ey = np.zeros((Nx, Ny + 1, Nz), dtype = np.float32)
    Ez = np.zeros((Nx, Ny, Nz + 1), dtype = np.float32)

    Hx = np.zeros((Nx, Ny + 1, Nz + 1), dtype = np.float32)
    Hy = np.zeros((Nx + 1, Ny, Nz + 1), dtype = np.float32)
    Hz = np.zeros((Nx + 1, Ny + 1, Nz), dtype = np.float32)

    #PML parameters
    pml_thickness = 20
    sigma_max = (3 + 1) * epsilon0 * c0 / (2 * dx)
    sigma_x, sigma_y, sigma_z = pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz)


def sigma_profile(sigma_max, pml_thickness, distance):
    return sigma_max * (distance / pml_thickness)**3

def pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz):
    sigma_x = np.zeros(Nx)
    sigma_y = np.zeros(Ny)
    sigma_z = np.zeros(Nz)
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
                    sigma_x, sigma_y, sigma_z, epsilon, mu,
                    dt, dx, dy, dz):
    Dx[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Dx[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * (
                    (Hz[1:-1, 1:, 1:-1] - Hz[1:-1, :-1, 1:-1]) / dy -
                    (Hy[1:-1, 1:-1, 1:] - Hy[1:-1, 1:-1, :-1]) / dz
            )
    )

    Dy[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * Dy[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * (
                    (Hx[1:-1, 1:-1, 1:] - Hx[1:-1, 1:-1, :-1]) / dz -
                    (Hz[1:, 1:-1, 1:-1] - Hz[:-1, 1:-1, 1:-1]) / dx
            )
    )

    Dz[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Dz[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * (
                    (Hy[1:, 1:-1, 1:-1] - Hy[:-1, 1:-1, 1:-1]) / dx -
                    (Hx[1:-1, 1:, 1:-1] - Hx[1:-1, :-1, 1:-1]) / dy
            )
    )

    Bx[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Bx[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * (
                    (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
                    (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
            )
    )

    By[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * By[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * (
                    (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
                    (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
            )
    )

    Bz[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Bz[1:-1, 1:-1, 1:-1] +
            dt / (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * (
                    (Ex[1:-1, 1:, 1:-1] - Ex[1:-1, :-1, 1:-1]) / dy -
                    (Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]) / dx
            )
    )

    Ex[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * Ex[1:-1, 1:-1, 1:-1] +
            1 / (epsilon[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1])) * (
                    (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Dx[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Dx[1:-1, 1:-1, 1:-1]
            )
    )

    Ey[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Ey[1:-1, 1:-1, 1:-1] +
            1 / (epsilon[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1])) * (
                    (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Dy[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Dy[0:0]
            )
    )

    Ez[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Ez[1:-1, 1:-1, 1:-1] +
            1 / (epsilon[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_y[0:0])) * (
                    (1 + dt / 2 * sigma_z[0:0]) * Dz[0:0] -
                    (1 - dt / 2 * sigma_z[0:0]) * Dz[0:0]
            )
    )

    Hx[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1]) * Hx[1:-1, 1:-1, 1:-1] +
            1 / (mu[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_z[1:-1, 1:-1, 1:-1])) * (
                    (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Bx[1:-1, 1:-1, 1:-1] -
                    (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Bx[1:-1, 1:-1, 1:-1]
            )
    )

    Hy[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_x[1:-1, 1:-1, 1:-1]) * Hy[1:-1, 1:-1, 1:-1] +
            1 / (mu[0:0] * (1 + dt / 2 * sigma_x[0:0])) * (
                    (1 + dt / 2 * sigma_y[0:0]) * By[0:0] -
                    (1 - dt / 2 * sigma_y[0:0]) * By[0:0]
            )
    )

    Hz[1:-1, 1:-1, 1:-1] = (
            (1 - dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) /
            (1 + dt / 2 * sigma_y[1:-1, 1:-1, 1:-1]) * Hz[1:-1, 1:-1, 1:-1] +
            1 / (mu[0:0] * (1 + dt / 2 * sigma_y[0:0])) * (
                    (1 + dt / 2 * sigma_z[0:0]) * Bz[0:0] -
                    (1 - dt / 2 * sigma_z[0:0]) * Bz[0:0]
            )
    )

    return Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz