#from backend import use_gpu
from dataclasses import dataclass

import cupy as cp

@dataclass
class FDTDAlignedParams_with_pec:
    '''This structure is used to store the parameters for the FDTD algorithm. Please Call this one as it contains the pec structure.'''
    sigma_y_Dx: any
    sigma_z_Dy: any
    sigma_x_Dz: any
    sigma_z_Ex: any
    sigma_x_Ex: any
    epsilon_Ex: any
    sigma_x_Ey: any
    sigma_y_Ey: any
    epsilon_Ey: any
    sigma_y_Ez: any
    sigma_z_Ez: any
    epsilon_Ez: any
    sigma_y_Bx: any
    sigma_z_By: any
    sigma_x_Bz: any
    sigma_z_Hx: any
    sigma_x_Hx: any
    mu_Hx: any
    sigma_x_Hy: any
    sigma_y_Hy: any
    mu_Hy: any
    sigma_y_Hz: any
    sigma_z_Hz: any
    mu_Hz: any
    dipole_pec_Ex: any
    dipole_pec_Ey: any
    dipole_pec_Ez: any

@dataclass
class FDTDAlignedParams:
    '''Old version without PEC. If you have a better PEC structure you can call this.'''
    sigma_y_Dx: any
    sigma_z_Dy: any
    sigma_x_Dz: any
    sigma_z_Ex: any
    sigma_x_Ex: any
    epsilon_Ex: any
    sigma_x_Ey: any
    sigma_y_Ey: any
    epsilon_Ey: any
    sigma_y_Ez: any
    sigma_z_Ez: any
    epsilon_Ez: any
    sigma_y_Bx: any
    sigma_z_By: any
    sigma_x_Bz: any
    sigma_z_Hx: any
    sigma_x_Hx: any
    mu_Hx: any
    sigma_x_Hy: any
    sigma_y_Hy: any
    mu_Hy: any
    sigma_y_Hz: any
    sigma_z_Hz: any
    mu_Hz: any

class fdtd_functions:
    '''This class contains the functions used in the FDTD algorithm.'''
    @staticmethod
    def gaussian_source(n, dt, sigma, omega0):
        '''Gaussian sine source function.'''
        t_now = (n - 0.5) * dt
        t0 = 4 * sigma
        return cp.exp(-((t_now - t0) / sigma) ** 2) * cp.sin(omega0 * (t_now - t0))

    @staticmethod
    def gprmax_gaussian_source(n, dt, omega0):
        '''Gaussian source function.'''
        t_now = (n - 0.5) * dt
        f_c = omega0 / (2 * cp.pi)
        zeta = 2 * cp.pi ** 2 * f_c ** 2
        chi = 1 / f_c
        return cp.exp(-zeta * (t_now - chi) ** 2)

    @staticmethod
    def sigma_profile(sigma_max, pml_thickness, distance):
        '''PML SIGMA distribution'''
        return sigma_max * (distance / pml_thickness) ** 3

    @staticmethod
    def pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz):
        '''PML profile function. put pml satisfy the yee grid'''
        sigma_x = cp.zeros(Nx)
        sigma_y = cp.zeros(Ny)
        sigma_z = cp.zeros(Nz)
        for i in range(pml_thickness):
            sigma_x[i] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
            sigma_x[-1 - i] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
        for j in range(pml_thickness):
            sigma_y[j] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
            sigma_y[-1 - j] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
        for k in range(pml_thickness):
            sigma_z[k] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - k)
            sigma_z[-1 - k] = fdtd_functions.sigma_profile(sigma_max, pml_thickness, pml_thickness - k)
        return sigma_x, sigma_y, sigma_z

    @staticmethod
    def fdtd_param_alignment(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu):
        '''This function is used to align the parameters of the FDTD algorithm. Make all the parameters has the shape satisfy the yee grid '''
        # Dx
        sigma_y_Dx = cp.zeros_like(Dx, dtype = cp.float64)
        sigma_y_Dx[1:-1, :, :] = 0.5 * (sigma_y_3d[:-1, :, :] + sigma_y_3d[1:, :, :])
        sigma_y_Dx[0, :, :] = sigma_y_3d[0, :, :]
        sigma_y_Dx[-1, :, :] = sigma_y_3d[-1, :, :]
        # Dy
        sigma_z_Dy = cp.zeros_like(Dy, dtype = cp.float64)
        sigma_z_Dy[:, 1:-1, :] = 0.5 * (sigma_z_3d[:, :-1, :] + sigma_z_3d[:, 1:, :])
        sigma_z_Dy[:, 0, :] = sigma_z_3d[:, 0, :]
        sigma_z_Dy[:, -1, :] = sigma_z_3d[:, -1, :]

        # Dz
        sigma_x_Dz = cp.zeros_like(Dz, dtype = cp.float64)
        sigma_x_Dz[:, :, 1:-1] = 0.5 * (sigma_x_3d[:, :, :-1] + sigma_x_3d[:, :, 1:])
        sigma_x_Dz[:, :, 0] = sigma_x_3d[:, :, 0]
        sigma_x_Dz[:, :, -1] = sigma_x_3d[:, :, -1]

        # Ex
        sigma_z_Ex = cp.zeros_like(Ex, dtype = cp.float64)
        sigma_z_Ex[1:-1, :, :] = 0.5 * (sigma_z_3d[:-1, :, :] + sigma_z_3d[1:, :, :])
        sigma_z_Ex[0, :, :] = sigma_z_3d[0, :, :]
        sigma_z_Ex[-1, :, :] = sigma_z_3d[-1, :, :]

        sigma_x_Ex = cp.zeros_like(Ex, dtype = cp.float64)
        sigma_x_Ex[1:-1, :, :] = 0.5 * (sigma_x_3d[:-1, :, :] + sigma_x_3d[1:, :, :])
        sigma_x_Ex[0, :, :] = sigma_x_3d[0, :, :]
        sigma_x_Ex[-1, :, :] = sigma_x_3d[-1, :, :]

        epsilon_Ex = cp.zeros_like(Ex, dtype = cp.float64)
        epsilon_Ex[1:-1, :, :] = 0.5 * (epsilon[:-1, :, :] + epsilon[1:, :, :])
        epsilon_Ex[0, :, :] = epsilon[0, :, :]
        epsilon_Ex[-1, :, :] = epsilon[-1, :, :]

        # Ey
        sigma_x_Ey = cp.zeros_like(Ey, dtype = cp.float64)
        sigma_x_Ey[:, 1:-1, :] = 0.5 * (sigma_x_3d[:, :-1, :] + sigma_x_3d[:, 1:, :])
        sigma_x_Ey[:, 0, :] = sigma_x_3d[:, 0, :]
        sigma_x_Ey[:, -1, :] = sigma_x_3d[:, -1, :]

        sigma_y_Ey = cp.zeros_like(Ey, dtype = cp.float64)
        sigma_y_Ey[:, 1:-1, :] = 0.5 * (sigma_y_3d[:, :-1, :] + sigma_y_3d[:, 1:, :])
        sigma_y_Ey[:, 0, :] = sigma_y_3d[:, 0, :]
        sigma_y_Ey[:, -1, :] = sigma_y_3d[:, -1, :]

        epsilon_Ey = cp.zeros_like(Ey, dtype = cp.float64)
        epsilon_Ey[:, 1:-1, :] = 0.5 * (epsilon[:, :-1, :] + epsilon[:, 1:, :])
        epsilon_Ey[:, 0, :] = epsilon[:, 0, :]
        epsilon_Ey[:, -1, :] = epsilon[:, -1, :]

        # Ez
        sigma_y_Ez = cp.zeros_like(Ez, dtype = cp.float64)
        sigma_y_Ez[:, :, 1:-1] = 0.5 * (sigma_y_3d[:, :, :-1] + sigma_y_3d[:, :, 1:])
        sigma_y_Ez[:, :, 0] = sigma_y_3d[:, :, 0]
        sigma_y_Ez[:, :, -1] = sigma_y_3d[:, :, -1]

        sigma_z_Ez = cp.zeros_like(Ez, dtype = cp.float64)
        sigma_z_Ez[:, :, 1:-1] = 0.5 * (sigma_z_3d[:, :, :-1] + sigma_z_3d[:, :, 1:])
        sigma_z_Ez[:, :, 0] = sigma_z_3d[:, :, 0]
        sigma_z_Ez[:, :, -1] = sigma_z_3d[:, :, -1]

        epsilon_Ez = cp.zeros_like(Ez, dtype = cp.float64)
        epsilon_Ez[:, :, 1:-1] = 0.5 * (epsilon[:, :, :-1] + epsilon[:, :, 1:])
        epsilon_Ez[:, :, 0] = epsilon[:, :, 0]
        epsilon_Ez[:, :, -1] = epsilon[:, :, -1]
        # Bx
        sigma_y_Bx = cp.zeros_like(Bx, dtype = cp.float64)
        sigma_y_Bx[:, 1:-1, 1:-1] = 0.25 * (
                sigma_y_3d[:, :-1, :-1] + sigma_y_3d[:, 1:, :-1] + sigma_y_3d[:, :-1,
                                                                   1:] + sigma_y_3d[:, 1:, 1:])
        sigma_y_Bx[:, 0, :] = sigma_y_Bx[:, 1, :]
        sigma_y_Bx[:, -1, :] = sigma_y_Bx[:, -2, :]
        sigma_y_Bx[:, :, 0] = sigma_y_Bx[:, :, 1]
        sigma_y_Bx[:, :, -1] = sigma_y_Bx[:, :, -2]
        # By
        sigma_z_By = cp.zeros_like(By, dtype = cp.float64)
        sigma_z_By[1:-1, :, 1:-1] = 0.25 * (
                sigma_z_3d[:-1, :, :-1] + sigma_z_3d[1:, :, :-1] + sigma_z_3d[:-1, :,
                                                                   1:] + sigma_z_3d[1:, :, 1:])
        sigma_z_By[0, :, :] = sigma_z_By[1, :, :]
        sigma_z_By[-1, :, :] = sigma_z_By[-2, :, :]
        sigma_z_By[:, :, 0] = sigma_z_By[:, :, 1]
        sigma_z_By[:, :, -1] = sigma_z_By[:, :, -2]

        # Bz
        sigma_x_Bz = cp.zeros_like(Bz, dtype = cp.float64)
        sigma_x_Bz[1:-1, 1:-1, :] = 0.25 * (
                sigma_x_3d[:-1, :-1, :] + sigma_x_3d[1:, :-1, :] + sigma_x_3d[:-1, 1:,
                                                                   :] + sigma_x_3d[1:, 1:, :])
        sigma_x_Bz[0, :, :] = sigma_x_Bz[1, :, :]
        sigma_x_Bz[-1, :, :] = sigma_x_Bz[-2, :, :]
        sigma_x_Bz[:, 0, :] = sigma_x_Bz[:, 1, :]
        sigma_x_Bz[:, -1, :] = sigma_x_Bz[:, -2, :]

        # Hx
        sigma_z_Hx = cp.zeros_like(Hx, dtype = cp.float64)
        sigma_z_Hx[:, 1:-1, 1:-1] = 0.25 * (
                    sigma_z_3d[:, :-1, :-1] + sigma_z_3d[:, 1:, :-1] + sigma_z_3d[:, :-1,
                                                                       1:] + sigma_z_3d[:, 1:, 1:])
        sigma_z_Hx[:, 0, :] = sigma_z_Hx[:, 1, :]
        sigma_z_Hx[:, -1, :] = sigma_z_Hx[:, -2, :]
        sigma_z_Hx[:, :, 0] = sigma_z_Hx[:, :, 1]
        sigma_z_Hx[:, :, -1] = sigma_z_Hx[:, :, -2]

        sigma_x_Hx = cp.zeros_like(Hx, dtype = cp.float64)
        sigma_x_Hx[:, 1:-1, 1:-1] = 0.25 * (
                    sigma_x_3d[:, :-1, :-1] + sigma_x_3d[:, 1:, :-1] + sigma_x_3d[:, :-1,
                                                                       1:] + sigma_x_3d[:, 1:, 1:])
        sigma_x_Hx[:, 0, :] = sigma_x_Hx[:, 1, :]
        sigma_x_Hx[:, -1, :] = sigma_x_Hx[:, -2, :]
        sigma_x_Hx[:, :, 0] = sigma_x_Hx[:, :, 1]
        sigma_x_Hx[:, :, -1] = sigma_x_Hx[:, :, -2]
        mu_Hx = cp.zeros_like(Hx, dtype = cp.float64)
        mu_Hx[:, 1:-1, 1:-1] = 0.25 * (
                    mu[:, :-1, :-1] + mu[:, 1:, :-1] + mu[:, :-1, 1:] + mu[:, 1:, 1:])
        mu_Hx[:, 0, :] = mu_Hx[:, 1, :]
        mu_Hx[:, -1, :] = mu_Hx[:, -2, :]
        mu_Hx[:, :, 0] = mu_Hx[:, :, 1]
        mu_Hx[:, :, -1] = mu_Hx[:, :, -2]

        # Hy
        sigma_x_Hy = cp.zeros_like(Hy, dtype = cp.float64)
        sigma_x_Hy[1:-1, :, 1:-1] = 0.25 * (
                    sigma_x_3d[:-1, :, :-1] + sigma_x_3d[1:, :, :-1] + sigma_x_3d[:-1, :,
                                                                       1:] + sigma_x_3d[1:, :, 1:])
        sigma_x_Hy[0, :, :] = sigma_x_Hy[1, :, :]
        sigma_x_Hy[-1, :, :] = sigma_x_Hy[-2, :, :]
        sigma_x_Hy[:, :, 0] = sigma_x_Hy[:, :, 1]
        sigma_x_Hy[:, :, -1] = sigma_x_Hy[:, :, -2]

        sigma_y_Hy = cp.zeros_like(Hy, dtype = cp.float64)
        sigma_y_Hy[1:-1, :, 1:-1] = 0.25 * (
                    sigma_y_3d[:-1, :, :-1] + sigma_y_3d[1:, :, :-1] + sigma_y_3d[:-1, :,
                                                                       1:] + sigma_y_3d[1:, :, 1:])
        sigma_y_Hy[0, :, :] = sigma_y_Hy[1, :, :]
        sigma_y_Hy[-1, :, :] = sigma_y_Hy[-2, :, :]
        sigma_y_Hy[:, :, 0] = sigma_y_Hy[:, :, 1]
        sigma_y_Hy[:, :, -1] = sigma_y_Hy[:, :, -2]

        mu_Hy = cp.zeros_like(Hy, dtype = cp.float64)
        mu_Hy[1:-1, :, 1:-1] = 0.25 * (
                    mu[:-1, :, :-1] + mu[1:, :, :-1] + mu[:-1, :, 1:] + mu[1:, :, 1:])
        mu_Hy[0, :, :] = mu_Hy[1, :, :]
        mu_Hy[-1, :, :] = mu_Hy[-2, :, :]
        mu_Hy[:, :, 0] = mu_Hy[:, :, 1]
        mu_Hy[:, :, -1] = mu_Hy[:, :, -2]
        # Hz
        sigma_y_Hz = cp.zeros_like(Hz, dtype = cp.float64)
        sigma_y_Hz[1:-1, 1:-1, :] = 0.25 * (
                    sigma_y_3d[:-1, :-1, :] + sigma_y_3d[1:, :-1, :] + sigma_y_3d[:-1, 1:,
                                                                       :] + sigma_y_3d[1:, 1:, :])
        sigma_y_Hz[:, 0, :] = sigma_y_Hz[:, 1, :]
        sigma_y_Hz[:, -1, :] = sigma_y_Hz[:, -2, :]
        sigma_y_Hz[0, :, :] = sigma_y_Hz[1, :, :]
        sigma_y_Hz[-1, :, :] = sigma_y_Hz[-2, :, :]

        sigma_z_Hz = cp.zeros_like(Hz, dtype = cp.float64)
        sigma_z_Hz[1:-1, 1:-1, :] = 0.25 * (
                    sigma_z_3d[:-1, :-1, :] + sigma_z_3d[1:, :-1, :] + sigma_z_3d[:-1, 1:,
                                                                       :] + sigma_z_3d[1:, 1:, :])
        sigma_z_Hz[:, 0, :] = sigma_z_Hz[:, 1, :]
        sigma_z_Hz[:, -1, :] = sigma_z_Hz[:, -2, :]
        sigma_z_Hz[0, :, :] = sigma_z_Hz[1, :, :]
        sigma_z_Hz[-1, :, :] = sigma_z_Hz[-2, :, :]

        mu_Hz = cp.zeros_like(Hz, dtype = cp.float64)
        mu_Hz[1:-1, 1:-1, :] = 0.25 * (
                    mu[:-1, :-1, :] + mu[1:, :-1, :] + mu[:-1, 1:, :] + mu[1:, 1:, :])
        mu_Hz[:, 0, :] = mu_Hz[:, 1, :]
        mu_Hz[:, -1, :] = mu_Hz[:, -2, :]
        mu_Hz[0, :, :] = mu_Hz[1, :, :]
        mu_Hz[-1, :, :] = mu_Hz[-2, :, :]

        return FDTDAlignedParams(
            sigma_y_Dx, sigma_z_Dy, sigma_x_Dz,
            sigma_z_Ex, sigma_x_Ex, epsilon_Ex,
            sigma_x_Ey, sigma_y_Ey, epsilon_Ey,
            sigma_y_Ez, sigma_z_Ez, epsilon_Ez,
            sigma_y_Bx, sigma_z_By, sigma_x_Bz,
            sigma_z_Hx, sigma_x_Hx, mu_Hx,
            sigma_x_Hy, sigma_y_Hy, mu_Hy,
            sigma_y_Hz, sigma_z_Hz, mu_Hz
        )

    @staticmethod
    def update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                         Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                         params:FDTDAlignedParams,
                         dt, dx, dy, dz):
        #     #unpack params
        sigma_y_Dx = params.sigma_y_Dx
        sigma_z_Dy = params.sigma_z_Dy
        sigma_x_Dz = params.sigma_x_Dz

        sigma_z_Ex = params.sigma_z_Ex
        sigma_x_Ex = params.sigma_x_Ex
        epsilon_Ex = params.epsilon_Ex
        sigma_x_Ey = params.sigma_x_Ey
        sigma_y_Ey = params.sigma_y_Ey
        epsilon_Ey = params.epsilon_Ey
        sigma_y_Ez = params.sigma_y_Ez
        sigma_z_Ez = params.sigma_z_Ez
        epsilon_Ez = params.epsilon_Ez
        sigma_y_Bx = params.sigma_y_Bx
        sigma_z_By = params.sigma_z_By
        sigma_x_Bz = params.sigma_x_Bz
        sigma_z_Hx = params.sigma_z_Hx
        sigma_x_Hx = params.sigma_x_Hx
        mu_Hx = params.mu_Hx
        sigma_x_Hy = params.sigma_x_Hy
        sigma_y_Hy = params.sigma_y_Hy
        mu_Hy = params.mu_Hy
        sigma_y_Hz = params.sigma_y_Hz
        sigma_z_Hz = params.sigma_z_Hz
        mu_Hz = params.mu_Hz



        Dx = ((1 - dt / 2 * sigma_y_Dx) / (1 + dt / 2 * sigma_y_Dx)) * Dx + \
                               dt / (1 + dt / 2 * sigma_y_Dx) * (
                                       (Hz[:, 1:, :] - Hz[:, :-1, :]) / (dy) -
                                       (Hy[:, :, 1:] - Hy[:, :, :-1]) / (dz)
                               )

        Dy = ((1 - dt / 2 * sigma_z_Dy) / (1 + dt / 2 * sigma_z_Dy)) * Dy + \
                               dt / (1 + dt / 2 * sigma_z_Dy) * (
                                       (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz -
                                       (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
                               )


        Dz = ((1 - dt / 2 * sigma_x_Dz) / (1 + dt / 2 * sigma_x_Dz)) * Dz + \
                               dt / (1 + dt / 2 * sigma_x_Dz) * (
                                       (Hy[1:, :, :] - Hy[:-1, :, :]) / dx -
                                       (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
                               )



        Ex = ((1 - dt / 2 * sigma_z_Ex) / (1 + dt / 2 * sigma_z_Ex)) * Ex + \
                               1 / (epsilon_Ex * (1 + dt / 2 * sigma_z_Ex)) * \
                               ((1 + dt / 2 * sigma_x_Ex) * Dx - (
                                       1 - dt / 2 * sigma_x_Ex) * Dx_old)



        Ey = ((1 - dt / 2 * sigma_x_Ey) / (1 + dt / 2 * sigma_x_Ey)) * Ey + \
                               1 / (epsilon_Ey * (1 + dt / 2 * sigma_x_Ey)) * \
                               ((1 + dt / 2 * sigma_y_Ey) * Dy - (
                                       1 - dt / 2 * sigma_y_Ey) * Dy_old)



        Ez = ((1 - dt / 2 * sigma_y_Ez) / (1 + dt / 2 * sigma_y_Ez)) * Ez + \
                               1 / (epsilon_Ez * (1 + dt / 2 * sigma_y_Ez)) * \
                               ((1 + dt / 2 * sigma_z_Ez) * Dz - (
                                       1 - dt / 2 * sigma_z_Ez) * Dz_old)


        Bx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1])) * Bx[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1]) * (
                                       (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
                                       (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
                               )



        By[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1])) * By[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1]) * (
                                       (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
                                       (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
                               )


        Bz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1])) * Bz[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1]) * (
                                       (Ex[1:-1, 1:, 1:-1] - Ex[1:-1, :-1, 1:-1]) / dy -
                                       (Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]) / dx
                               )



        Hx[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1])) * Hx[1:-1, 1:-1,
                                                                                1:-1] +
                1 / (mu_Hx[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_x_Hx[1:-1, 1:-1, 1:-1]) * Bx[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_x_Hx[1:-1, 1:-1, 1:-1]) * Bx_old[1:-1, 1:-1, 1:-1]
                )
        )


        Hy[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1])) * Hy[1:-1, 1:-1,
                                                                                1:-1] +
                1 / (mu_Hy[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_y_Hy[1:-1, 1:-1, 1:-1]) * By[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_y_Hy[1:-1, 1:-1, 1:-1]) * By_old[1:-1, 1:-1, 1:-1]
                )
        )


        Hz[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1]) / (1 + dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1])) * Hz[1:-1, 1:-1,
                                                                                1:-1] +
                1 / (mu_Hz[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_z_Hz[1:-1, 1:-1, 1:-1]) * Bz[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_z_Hz[1:-1, 1:-1, 1:-1]) * Bz_old[1:-1, 1:-1, 1:-1]
                )
        )

        Dx_old = cp.copy(Dx)
        Dy_old = cp.copy(Dy)
        Dz_old = cp.copy(Dz)
        Bx_old = cp.copy(Bx)
        By_old = cp.copy(By)
        Bz_old = cp.copy(Bz)

        return Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old

    @staticmethod
    def fdtd_param_alignment_with_pec(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, sigma_x_3d, sigma_y_3d,
                             sigma_z_3d, epsilon, mu,
                             i_x_src, i_y_src, i_z_src, i_z_dipole_start, i_z_dipole_end):
        '''Same as before, but with PEC boundary condition.'''
        # Dx
        sigma_y_Dx = cp.zeros_like(Dx, dtype = cp.float32)
        sigma_y_Dx[1:-1, :, :] = 0.5 * (sigma_y_3d[:-1, :, :] + sigma_y_3d[1:, :, :])
        sigma_y_Dx[0, :, :] = sigma_y_3d[0, :, :]
        sigma_y_Dx[-1, :, :] = sigma_y_3d[-1, :, :]
        # Dy
        sigma_z_Dy = cp.zeros_like(Dy, dtype = cp.float32)
        sigma_z_Dy[:, 1:-1, :] = 0.5 * (sigma_z_3d[:, :-1, :] + sigma_z_3d[:, 1:, :])
        sigma_z_Dy[:, 0, :] = sigma_z_3d[:, 0, :]
        sigma_z_Dy[:, -1, :] = sigma_z_3d[:, -1, :]

        # Dz
        sigma_x_Dz = cp.zeros_like(Dz, dtype = cp.float32)
        sigma_x_Dz[:, :, 1:-1] = 0.5 * (sigma_x_3d[:, :, :-1] + sigma_x_3d[:, :, 1:])
        sigma_x_Dz[:, :, 0] = sigma_x_3d[:, :, 0]
        sigma_x_Dz[:, :, -1] = sigma_x_3d[:, :, -1]

        # Ex
        sigma_z_Ex = cp.zeros_like(Ex, dtype = cp.float32)
        sigma_z_Ex[1:-1, :, :] = 0.5 * (sigma_z_3d[:-1, :, :] + sigma_z_3d[1:, :, :])
        sigma_z_Ex[0, :, :] = sigma_z_3d[0, :, :]
        sigma_z_Ex[-1, :, :] = sigma_z_3d[-1, :, :]

        sigma_x_Ex = cp.zeros_like(Ex, dtype = cp.float32)
        sigma_x_Ex[1:-1, :, :] = 0.5 * (sigma_x_3d[:-1, :, :] + sigma_x_3d[1:, :, :])
        sigma_x_Ex[0, :, :] = sigma_x_3d[0, :, :]
        sigma_x_Ex[-1, :, :] = sigma_x_3d[-1, :, :]

        epsilon_Ex = cp.zeros_like(Ex, dtype = cp.float32)
        epsilon_Ex[1:-1, :, :] = 0.5 * (epsilon[:-1, :, :] + epsilon[1:, :, :])
        epsilon_Ex[0, :, :] = epsilon[0, :, :]
        epsilon_Ex[-1, :, :] = epsilon[-1, :, :]

        # Ey
        sigma_x_Ey = cp.zeros_like(Ey, dtype = cp.float32)
        sigma_x_Ey[:, 1:-1, :] = 0.5 * (sigma_x_3d[:, :-1, :] + sigma_x_3d[:, 1:, :])
        sigma_x_Ey[:, 0, :] = sigma_x_3d[:, 0, :]
        sigma_x_Ey[:, -1, :] = sigma_x_3d[:, -1, :]

        sigma_y_Ey = cp.zeros_like(Ey, dtype = cp.float32)
        sigma_y_Ey[:, 1:-1, :] = 0.5 * (sigma_y_3d[:, :-1, :] + sigma_y_3d[:, 1:, :])
        sigma_y_Ey[:, 0, :] = sigma_y_3d[:, 0, :]
        sigma_y_Ey[:, -1, :] = sigma_y_3d[:, -1, :]

        epsilon_Ey = cp.zeros_like(Ey, dtype = cp.float32)
        epsilon_Ey[:, 1:-1, :] = 0.5 * (epsilon[:, :-1, :] + epsilon[:, 1:, :])
        epsilon_Ey[:, 0, :] = epsilon[:, 0, :]
        epsilon_Ey[:, -1, :] = epsilon[:, -1, :]

        # Ez
        sigma_y_Ez = cp.zeros_like(Ez, dtype = cp.float32)
        sigma_y_Ez[:, :, 1:-1] = 0.5 * (sigma_y_3d[:, :, :-1] + sigma_y_3d[:, :, 1:])
        sigma_y_Ez[:, :, 0] = sigma_y_3d[:, :, 0]
        sigma_y_Ez[:, :, -1] = sigma_y_3d[:, :, -1]

        sigma_z_Ez = cp.zeros_like(Ez, dtype = cp.float32)
        sigma_z_Ez[:, :, 1:-1] = 0.5 * (sigma_z_3d[:, :, :-1] + sigma_z_3d[:, :, 1:])
        sigma_z_Ez[:, :, 0] = sigma_z_3d[:, :, 0]
        sigma_z_Ez[:, :, -1] = sigma_z_3d[:, :, -1]

        epsilon_Ez = cp.zeros_like(Ez, dtype = cp.float32)
        epsilon_Ez[:, :, 1:-1] = 0.5 * (epsilon[:, :, :-1] + epsilon[:, :, 1:])
        epsilon_Ez[:, :, 0] = epsilon[:, :, 0]
        epsilon_Ez[:, :, -1] = epsilon[:, :, -1]
        # Bx
        sigma_y_Bx = cp.zeros_like(Bx, dtype = cp.float32)
        sigma_y_Bx[:, 1:-1, 1:-1] = 0.25 * (
                sigma_y_3d[:, :-1, :-1] + sigma_y_3d[:, 1:, :-1] + sigma_y_3d[:, :-1,
                                                                   1:] + sigma_y_3d[:, 1:, 1:])
        sigma_y_Bx[:, 0, :] = sigma_y_Bx[:, 1, :]
        sigma_y_Bx[:, -1, :] = sigma_y_Bx[:, -2, :]
        sigma_y_Bx[:, :, 0] = sigma_y_Bx[:, :, 1]
        sigma_y_Bx[:, :, -1] = sigma_y_Bx[:, :, -2]
        # By
        sigma_z_By = cp.zeros_like(By, dtype = cp.float32)
        sigma_z_By[1:-1, :, 1:-1] = 0.25 * (
                sigma_z_3d[:-1, :, :-1] + sigma_z_3d[1:, :, :-1] + sigma_z_3d[:-1, :,
                                                                   1:] + sigma_z_3d[1:, :, 1:])
        sigma_z_By[0, :, :] = sigma_z_By[1, :, :]
        sigma_z_By[-1, :, :] = sigma_z_By[-2, :, :]
        sigma_z_By[:, :, 0] = sigma_z_By[:, :, 1]
        sigma_z_By[:, :, -1] = sigma_z_By[:, :, -2]

        # Bz
        sigma_x_Bz = cp.zeros_like(Bz, dtype = cp.float32)
        sigma_x_Bz[1:-1, 1:-1, :] = 0.25 * (
                sigma_x_3d[:-1, :-1, :] + sigma_x_3d[1:, :-1, :] + sigma_x_3d[:-1, 1:,
                                                                   :] + sigma_x_3d[1:, 1:, :])
        sigma_x_Bz[0, :, :] = sigma_x_Bz[1, :, :]
        sigma_x_Bz[-1, :, :] = sigma_x_Bz[-2, :, :]
        sigma_x_Bz[:, 0, :] = sigma_x_Bz[:, 1, :]
        sigma_x_Bz[:, -1, :] = sigma_x_Bz[:, -2, :]

        # Hx
        sigma_z_Hx = cp.zeros_like(Hx, dtype = cp.float32)
        sigma_z_Hx[:, 1:-1, 1:-1] = 0.25 * (
                sigma_z_3d[:, :-1, :-1] + sigma_z_3d[:, 1:, :-1] + sigma_z_3d[:, :-1,
                                                                   1:] + sigma_z_3d[:, 1:, 1:])
        sigma_z_Hx[:, 0, :] = sigma_z_Hx[:, 1, :]
        sigma_z_Hx[:, -1, :] = sigma_z_Hx[:, -2, :]
        sigma_z_Hx[:, :, 0] = sigma_z_Hx[:, :, 1]
        sigma_z_Hx[:, :, -1] = sigma_z_Hx[:, :, -2]

        sigma_x_Hx = cp.zeros_like(Hx, dtype = cp.float32)
        sigma_x_Hx[:, 1:-1, 1:-1] = 0.25 * (
                sigma_x_3d[:, :-1, :-1] + sigma_x_3d[:, 1:, :-1] + sigma_x_3d[:, :-1,
                                                                   1:] + sigma_x_3d[:, 1:, 1:])
        sigma_x_Hx[:, 0, :] = sigma_x_Hx[:, 1, :]
        sigma_x_Hx[:, -1, :] = sigma_x_Hx[:, -2, :]
        sigma_x_Hx[:, :, 0] = sigma_x_Hx[:, :, 1]
        sigma_x_Hx[:, :, -1] = sigma_x_Hx[:, :, -2]
        mu_Hx = cp.zeros_like(Hx, dtype = cp.float32)
        mu_Hx[:, 1:-1, 1:-1] = 0.25 * (
                mu[:, :-1, :-1] + mu[:, 1:, :-1] + mu[:, :-1, 1:] + mu[:, 1:, 1:])
        mu_Hx[:, 0, :] = mu_Hx[:, 1, :]
        mu_Hx[:, -1, :] = mu_Hx[:, -2, :]
        mu_Hx[:, :, 0] = mu_Hx[:, :, 1]
        mu_Hx[:, :, -1] = mu_Hx[:, :, -2]

        # Hy
        sigma_x_Hy = cp.zeros_like(Hy, dtype = cp.float32)
        sigma_x_Hy[1:-1, :, 1:-1] = 0.25 * (
                sigma_x_3d[:-1, :, :-1] + sigma_x_3d[1:, :, :-1] + sigma_x_3d[:-1, :,
                                                                   1:] + sigma_x_3d[1:, :, 1:])
        sigma_x_Hy[0, :, :] = sigma_x_Hy[1, :, :]
        sigma_x_Hy[-1, :, :] = sigma_x_Hy[-2, :, :]
        sigma_x_Hy[:, :, 0] = sigma_x_Hy[:, :, 1]
        sigma_x_Hy[:, :, -1] = sigma_x_Hy[:, :, -2]

        sigma_y_Hy = cp.zeros_like(Hy, dtype = cp.float32)
        sigma_y_Hy[1:-1, :, 1:-1] = 0.25 * (
                sigma_y_3d[:-1, :, :-1] + sigma_y_3d[1:, :, :-1] + sigma_y_3d[:-1, :,
                                                                   1:] + sigma_y_3d[1:, :, 1:])
        sigma_y_Hy[0, :, :] = sigma_y_Hy[1, :, :]
        sigma_y_Hy[-1, :, :] = sigma_y_Hy[-2, :, :]
        sigma_y_Hy[:, :, 0] = sigma_y_Hy[:, :, 1]
        sigma_y_Hy[:, :, -1] = sigma_y_Hy[:, :, -2]

        mu_Hy = cp.zeros_like(Hy, dtype = cp.float32)
        mu_Hy[1:-1, :, 1:-1] = 0.25 * (
                mu[:-1, :, :-1] + mu[1:, :, :-1] + mu[:-1, :, 1:] + mu[1:, :, 1:])
        mu_Hy[0, :, :] = mu_Hy[1, :, :]
        mu_Hy[-1, :, :] = mu_Hy[-2, :, :]
        mu_Hy[:, :, 0] = mu_Hy[:, :, 1]
        mu_Hy[:, :, -1] = mu_Hy[:, :, -2]
        # Hz
        sigma_y_Hz = cp.zeros_like(Hz, dtype = cp.float32)
        sigma_y_Hz[1:-1, 1:-1, :] = 0.25 * (
                sigma_y_3d[:-1, :-1, :] + sigma_y_3d[1:, :-1, :] + sigma_y_3d[:-1, 1:,
                                                                   :] + sigma_y_3d[1:, 1:, :])
        sigma_y_Hz[:, 0, :] = sigma_y_Hz[:, 1, :]
        sigma_y_Hz[:, -1, :] = sigma_y_Hz[:, -2, :]
        sigma_y_Hz[0, :, :] = sigma_y_Hz[1, :, :]
        sigma_y_Hz[-1, :, :] = sigma_y_Hz[-2, :, :]

        sigma_z_Hz = cp.zeros_like(Hz, dtype = cp.float32)
        sigma_z_Hz[1:-1, 1:-1, :] = 0.25 * (
                sigma_z_3d[:-1, :-1, :] + sigma_z_3d[1:, :-1, :] + sigma_z_3d[:-1, 1:,
                                                                   :] + sigma_z_3d[1:, 1:, :])
        sigma_z_Hz[:, 0, :] = sigma_z_Hz[:, 1, :]
        sigma_z_Hz[:, -1, :] = sigma_z_Hz[:, -2, :]
        sigma_z_Hz[0, :, :] = sigma_z_Hz[1, :, :]
        sigma_z_Hz[-1, :, :] = sigma_z_Hz[-2, :, :]

        mu_Hz = cp.zeros_like(Hz, dtype = cp.float32)
        mu_Hz[1:-1, 1:-1, :] = 0.25 * (
                mu[:-1, :-1, :] + mu[1:, :-1, :] + mu[:-1, 1:, :] + mu[1:, 1:, :])
        mu_Hz[:, 0, :] = mu_Hz[:, 1, :]
        mu_Hz[:, -1, :] = mu_Hz[:, -2, :]
        mu_Hz[0, :, :] = mu_Hz[1, :, :]
        mu_Hz[-1, :, :] = mu_Hz[-2, :, :]

        pec = cp.ones_like(mu, dtype = cp.float32)
        pec[i_x_src, i_y_src, i_z_dipole_start:i_z_src] = 0
        pec[i_x_src, i_y_src, i_z_src + 1:i_z_dipole_end] = 0

        dipole_pec_Ex = cp.ones_like(Ex, dtype = cp.float32)
        dipole_pec_Ex[1:-1, :, :] = 0.5 * (pec[:-1, :, :] + pec[1:, :, :])
        dipole_pec_Ex[0, :, :] = pec[0, :, :]
        dipole_pec_Ex[-1, :, :] = pec[-1, :, :]

        dipole_pec_Ey = cp.ones_like(Ey, dtype = cp.float32)
        dipole_pec_Ey[:, 1:-1, :] = 0.5 * (pec[:, :-1, :] + pec[:, 1:, :])
        dipole_pec_Ey[:, 0, :] = pec[:, 0, :]
        dipole_pec_Ey[:, -1, :] = pec[:, -1, :]

        dipole_pec_Ez = cp.ones_like(Ez, dtype = cp.float32)
        dipole_pec_Ez[:, :, 1:-1] = 0.5 * (pec[:, :, :-1] + pec[:, :, 1:])
        dipole_pec_Ez[:, :, 0] = pec[:, :, 0]
        dipole_pec_Ez[:, :, -1] = pec[:, :, -1]

        # dipole_pec_Ex[dipole_pec_Ex != 1] = 0
        # dipole_pec_Ey[dipole_pec_Ey != 1] = 0
        # dipole_pec_Ez[dipole_pec_Ez != 1] = 0
        # # dipole_pec_Ez[i_x_src, i_y_src, i_z_src] = 1
        # dipole_pec_Ez[i_x_src, i_y_src, i_z_src] = 0.5
        # dipole_pec_Ez[i_x_src, i_y_src, i_z_src+1] = 0.5

        return FDTDAlignedParams_with_pec(
            sigma_y_Dx, sigma_z_Dy, sigma_x_Dz,
            sigma_z_Ex, sigma_x_Ex, epsilon_Ex,
            sigma_x_Ey, sigma_y_Ey, epsilon_Ey,
            sigma_y_Ez, sigma_z_Ez, epsilon_Ez,
            sigma_y_Bx, sigma_z_By, sigma_x_Bz,
            sigma_z_Hx, sigma_x_Hx, mu_Hx,
            sigma_x_Hy, sigma_y_Hy, mu_Hy,
            sigma_y_Hz, sigma_z_Hz, mu_Hz,
            dipole_pec_Ex, dipole_pec_Ey, dipole_pec_Ez
        )

    @staticmethod
    def update_equations_with_pec(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                         Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
                         params: FDTDAlignedParams,
                         dt, dx, dy, dz):
        '''same as before, but with PEC boundary condition.'''
        #     #unpack params
        sigma_y_Dx = params.sigma_y_Dx
        sigma_z_Dy = params.sigma_z_Dy
        sigma_x_Dz = params.sigma_x_Dz
        sigma_z_Ex = params.sigma_z_Ex
        sigma_x_Ex = params.sigma_x_Ex
        epsilon_Ex = params.epsilon_Ex
        sigma_x_Ey = params.sigma_x_Ey
        sigma_y_Ey = params.sigma_y_Ey
        epsilon_Ey = params.epsilon_Ey
        sigma_y_Ez = params.sigma_y_Ez
        sigma_z_Ez = params.sigma_z_Ez
        epsilon_Ez = params.epsilon_Ez
        sigma_y_Bx = params.sigma_y_Bx
        sigma_z_By = params.sigma_z_By
        sigma_x_Bz = params.sigma_x_Bz
        sigma_z_Hx = params.sigma_z_Hx
        sigma_x_Hx = params.sigma_x_Hx
        mu_Hx = params.mu_Hx
        sigma_x_Hy = params.sigma_x_Hy
        sigma_y_Hy = params.sigma_y_Hy
        mu_Hy = params.mu_Hy
        sigma_y_Hz = params.sigma_y_Hz
        sigma_z_Hz = params.sigma_z_Hz
        mu_Hz = params.mu_Hz
        dipole_pec_Ex = params.dipole_pec_Ex
        dipole_pec_Ey = params.dipole_pec_Ey
        dipole_pec_Ez = params.dipole_pec_Ez

        Bx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1]) / (
                    1 + dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1])) * Bx[1:-1,
                                                                  1:-1,
                                                                  1:-1] + \
                               dt / (1 + dt / 2 * sigma_y_Bx[1:-1, 1:-1, 1:-1]) * (
                                       (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
                                       (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
                               )

        By[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1]) / (
                    1 + dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1])) * By[1:-1,
                                                                  1:-1,
                                                                  1:-1] + \
                               dt / (1 + dt / 2 * sigma_z_By[1:-1, 1:-1, 1:-1]) * (
                                       (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
                                       (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
                               )

        Bz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1]) / (
                    1 + dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1])) * Bz[1:-1,
                                                                  1:-1,
                                                                  1:-1] + \
                               dt / (1 + dt / 2 * sigma_x_Bz[1:-1, 1:-1, 1:-1]) * (
                                       (Ex[1:-1, 1:, 1:-1] - Ex[1:-1, :-1, 1:-1]) / dy -
                                       (Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]) / dx
                               )

        Hx[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1]) / (
                            1 + dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1])) * Hx[1:-1, 1:-1,
                                                                          1:-1] +
                1 / (mu_Hx[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_z_Hx[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_x_Hx[1:-1, 1:-1, 1:-1]) * Bx[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_x_Hx[1:-1, 1:-1, 1:-1]) * Bx_old[1:-1, 1:-1, 1:-1]
                )
        )

        Hy[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1]) / (
                            1 + dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1])) * Hy[1:-1, 1:-1,
                                                                          1:-1] +
                1 / (mu_Hy[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_x_Hy[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_y_Hy[1:-1, 1:-1, 1:-1]) * By[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_y_Hy[1:-1, 1:-1, 1:-1]) * By_old[1:-1, 1:-1, 1:-1]
                )
        )

        Hz[1:-1, 1:-1, 1:-1] = (
                ((1 - dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1]) / (
                            1 + dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1])) * Hz[1:-1, 1:-1,
                                                                          1:-1] +
                1 / (mu_Hz[1:-1, 1:-1, 1:-1] * (1 + dt / 2 * sigma_y_Hz[1:-1, 1:-1, 1:-1])) *
                (
                        (1 + dt / 2 * sigma_z_Hz[1:-1, 1:-1, 1:-1]) * Bz[1:-1, 1:-1, 1:-1] -
                        (1 - dt / 2 * sigma_z_Hz[1:-1, 1:-1, 1:-1]) * Bz_old[1:-1, 1:-1, 1:-1]
                )
        )

        Dx = ((1 - dt / 2 * sigma_y_Dx) / (1 + dt / 2 * sigma_y_Dx)) * Dx + \
             dt / (1 + dt / 2 * sigma_y_Dx) * (
                     (Hz[:, 1:, :] - Hz[:, :-1, :]) / (dy) -
                     (Hy[:, :, 1:] - Hy[:, :, :-1]) / (dz)
             )

        Dy = ((1 - dt / 2 * sigma_z_Dy) / (1 + dt / 2 * sigma_z_Dy)) * Dy + \
             dt / (1 + dt / 2 * sigma_z_Dy) * (
                     (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz -
                     (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
             )

        Dz = ((1 - dt / 2 * sigma_x_Dz) / (1 + dt / 2 * sigma_x_Dz)) * Dz + \
             dt / (1 + dt / 2 * sigma_x_Dz) * (
                     (Hy[1:, :, :] - Hy[:-1, :, :]) / dx -
                     (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
             )

        Ex = ((1 - dt / 2 * sigma_z_Ex) / (1 + dt / 2 * sigma_z_Ex)) * Ex + \
             1 / (epsilon_Ex * (1 + dt / 2 * sigma_z_Ex)) * \
             ((1 + dt / 2 * sigma_x_Ex) * Dx - (
                     1 - dt / 2 * sigma_x_Ex) * Dx_old)

        Ey = ((1 - dt / 2 * sigma_x_Ey) / (1 + dt / 2 * sigma_x_Ey)) * Ey + \
             1 / (epsilon_Ey * (1 + dt / 2 * sigma_x_Ey)) * \
             ((1 + dt / 2 * sigma_y_Ey) * Dy - (
                     1 - dt / 2 * sigma_y_Ey) * Dy_old)

        Ez = ((1 - dt / 2 * sigma_y_Ez) / (1 + dt / 2 * sigma_y_Ez)) * Ez + \
             1 / (epsilon_Ez * (1 + dt / 2 * sigma_y_Ez)) * \
             ((1 + dt / 2 * sigma_z_Ez) * Dz - (
                     1 - dt / 2 * sigma_z_Ez) * Dz_old)
        Ex = dipole_pec_Ex * Ex
        Ey = dipole_pec_Ey * Ey
        Ez = dipole_pec_Ez * Ez

        Dx_old = cp.copy(Dx)
        Dy_old = cp.copy(Dy)
        Dz_old = cp.copy(Dz)
        Bx_old = cp.copy(Bx)
        By_old = cp.copy(By)
        Bz_old = cp.copy(Bz)

        return Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old

