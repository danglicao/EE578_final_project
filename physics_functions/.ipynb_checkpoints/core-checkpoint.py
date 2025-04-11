#from backend import use_gpu

import cupy as cp

class fdtd_functions:
    @staticmethod
    def gaussian_source(n, dt, sigma, omega0):
        t_now = (n - 0.5) * dt
        t0 = 4 * sigma
        return cp.exp(-((t_now - t0) / sigma) ** 2) * cp.sin(omega0 * (t_now - t0))

    @staticmethod
    def sigma_profile(sigma_max, pml_thickness, distance):
        return sigma_max * (distance / pml_thickness) ** 3

    @staticmethod
    def pml_profile(sigma_max, pml_thickness, Nx, Ny, Nz):
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
    def update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,params = None):
        # Unpack parameters
        if params is not None:
            Dx = params.Dx
            Dy = params.Dy
            Dz = params.Dz
            Ex = params.Ex
            Ey = params.Ey
            Ez = params.Ez
            Hx = params.Hx
            Hy = params.Hy
            Hz = params.Hz
            Bx = params.Bx
            By = params.By
            Bz = params.Bz

            Dx_old = params.Dx_old
            Dy_old = params.Dy_old
            Dz_old = params.Dz_old
            Bx_old = params.Bx_old
            By_old = params.By_old
            Bz_old = params.Bz_old

            sigma_x = params.sigma_x_3d
            sigma_y = params.sigma_y_3d
            sigma_z = params.sigma_z_3d
            epsilon = params.epsilon
            mu = params.mu

            dt = params.dt
            dx = params.dx
            dy = params.dy
            dz = params.dz

        sigma_y_Dx = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])

        sigma_y_Dx = sigma_y_Dx[:, 1:-1, 1:-1]

        Dx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Dx) / (1 + dt / 2 * sigma_y_Dx)) * Dx[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_y_Dx) * (
                                       (Hz[1:-1, 2:-1, 1:-1] - Hz[1:-1, 1:-2, 1:-1]) / (2 * dy) -
                                       (Hy[1:-1, 1:-1, 2:-1] - Hy[1:-1, 1:-1, 1:-2]) / (2 * dz)
                               )

        sigma_z_Dy = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
        sigma_z_Dy = sigma_z_Dy[1:-1, :, 1:-1]
        Dy[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_Dy) / (1 + dt / 2 * sigma_z_Dy)) * Dy[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_z_Dy) * (
                                       (Hx[1:-1, 1:-1, 2:-1] - Hx[1:-1, 1:-1, 1:-2]) / dz -
                                       (Hz[2:-1, 1:-1, 1:-1] - Hz[1:-2, 1:-1, 1:-1]) / dx
                               )

        sigma_x_Dz = 0.5 * (sigma_x[:, :, :-1] + sigma_x[:, :, 1:])
        sigma_x_Dz = sigma_x_Dz[1:-1, 1:-1, :]
        Dz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Dz) / (1 + dt / 2 * sigma_x_Dz)) * Dz[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_x_Dz) * (
                                       (Hy[2:-1, 1:-1, 1:-1] - Hy[1:-2, 1:-1, 1:-1]) / dx -
                                       (Hx[1:-1, 2:-1, 1:-1] - Hx[1:-1, 1:-2, 1:-1]) / dy
                               )

        sigma_z_Ex = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
        sigma_z_Ex = sigma_z_Ex[:, 1:-1, 1:-1]

        sigma_x_Ex = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
        sigma_x_Ex = sigma_x_Ex[:, 1:-1, 1:-1]

        epsilon_Ex = 0.5 * (epsilon[:-1, :, :] + epsilon[1:, :, :])
        epsilon_Ex = epsilon_Ex[:, 1:-1, 1:-1]

        Ex[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_Ex) / (1 + dt / 2 * sigma_z_Ex)) * Ex[1:-1,
                                                                                         1:-1,
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

        Ey[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Ey) / (1 + dt / 2 * sigma_x_Ey)) * Ey[1:-1,
                                                                                         1:-1,
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

        Ez[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Ez) / (1 + dt / 2 * sigma_y_Ez)) * Ez[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               1 / (epsilon_Ez * (1 + dt / 2 * sigma_y_Ez)) * \
                               ((1 + dt / 2 * sigma_z_Ez) * Dz[1:-1, 1:-1, 1:-1] - (
                                       1 - dt / 2 * sigma_z_Ez) * Dz_old[1:-1, 1:-1, 1:-1])

        sigma_y_Bx = 0.5 * (sigma_y[:, :-1, :] + sigma_y[:, 1:, :])
        sigma_y_Bx = 0.5 * (sigma_y_Bx[:, :, :-1] + sigma_y_Bx[:, :, 1:])
        sigma_y_Bx = sigma_y_Bx[1:-1, :, :]

        Bx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Bx) / (1 + dt / 2 * sigma_y_Bx)) * Bx[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_y_Bx) * (
                                       (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
                                       (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
                               )

        sigma_z_By = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
        sigma_z_By = 0.5 * (sigma_z_By[:, :, :-1] + sigma_z_By[:, :, 1:])
        sigma_z_By = sigma_z_By[:, 1:-1, :]

        By[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_By) / (1 + dt / 2 * sigma_z_By)) * By[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_z_By) * (
                                       (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
                                       (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
                               )

        sigma_x_Bz = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
        sigma_x_Bz = 0.5 * (sigma_x_Bz[:, :-1, :] + sigma_x_Bz[:, 1:, :])
        sigma_x_Bz = sigma_x_Bz[:, :, 1:-1]
        Bz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Bz) / (1 + dt / 2 * sigma_x_Bz)) * Bz[1:-1,
                                                                                         1:-1,
                                                                                         1:-1] + \
                               dt / (1 + dt / 2 * sigma_x_Bz) * (
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
                ((1 - dt / 2 * sigma_z_Hx_in) / (1 + dt / 2 * sigma_z_Hx_in)) * Hx[1:-1, 1:-1,
                                                                                1:-1] +
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
                ((1 - dt / 2 * sigma_x_Hy_in) / (1 + dt / 2 * sigma_x_Hy_in)) * Hy[1:-1, 1:-1,
                                                                                1:-1] +
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
                ((1 - dt / 2 * sigma_y_Hz_in) / (1 + dt / 2 * sigma_y_Hz_in)) * Hz[1:-1, 1:-1,
                                                                                1:-1] +
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
    # def update_equations(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
    #                      Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
    #                      sigma_x, sigma_y, sigma_z, epsilon, mu,
    #                      dt, dx, dy, dz):
    #
    #     sigma_y_Dx = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])
    #
    #     sigma_y_Dx = sigma_y_Dx[:, 1:-1, 1:-1]
    #
    #     Dx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Dx) / (1 + dt / 2 * sigma_y_Dx)) * Dx[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_y_Dx) * (
    #                                    (Hz[1:-1, 2:-1, 1:-1] - Hz[1:-1, 1:-2, 1:-1]) / (2 * dy) -
    #                                    (Hy[1:-1, 1:-1, 2:-1] - Hy[1:-1, 1:-1, 1:-2]) / (2 * dz)
    #                            )
    #
    #     sigma_z_Dy = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
    #     sigma_z_Dy = sigma_z_Dy[1:-1, :, 1:-1]
    #     Dy[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_Dy) / (1 + dt / 2 * sigma_z_Dy)) * Dy[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_z_Dy) * (
    #                                    (Hx[1:-1, 1:-1, 2:-1] - Hx[1:-1, 1:-1, 1:-2]) / dz -
    #                                    (Hz[2:-1, 1:-1, 1:-1] - Hz[1:-2, 1:-1, 1:-1]) / dx
    #                            )
    #
    #     sigma_x_Dz = 0.5 * (sigma_x[:, :, :-1] + sigma_x[:, :, 1:])
    #     sigma_x_Dz = sigma_x_Dz[1:-1, 1:-1, :]
    #     Dz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Dz) / (1 + dt / 2 * sigma_x_Dz)) * Dz[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_x_Dz) * (
    #                                    (Hy[2:-1, 1:-1, 1:-1] - Hy[1:-2, 1:-1, 1:-1]) / dx -
    #                                    (Hx[1:-1, 2:-1, 1:-1] - Hx[1:-1, 1:-2, 1:-1]) / dy
    #                            )
    #
    #     sigma_z_Ex = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    #     sigma_z_Ex = sigma_z_Ex[:, 1:-1, 1:-1]
    #
    #     sigma_x_Ex = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    #     sigma_x_Ex = sigma_x_Ex[:, 1:-1, 1:-1]
    #
    #     epsilon_Ex = 0.5 * (epsilon[:-1, :, :] + epsilon[1:, :, :])
    #     epsilon_Ex = epsilon_Ex[:, 1:-1, 1:-1]
    #
    #     Ex[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_Ex) / (1 + dt / 2 * sigma_z_Ex)) * Ex[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            1 / (epsilon_Ex * (1 + dt / 2 * sigma_z_Ex)) * \
    #                            ((1 + dt / 2 * sigma_x_Ex) * Dx[1:-1, 1:-1, 1:-1] - (
    #                                    1 - dt / 2 * sigma_x_Ex) * Dx_old[1:-1, 1:-1, 1:-1])
    #
    #     sigma_x_Ey = 0.5 * (sigma_x[:, :-1, :] + sigma_x[:, 1:, :])
    #     sigma_x_Ey = sigma_x_Ey[1:-1, :, 1:-1]
    #
    #     sigma_y_Ey = 0.5 * (sigma_y[:, :-1, :] + sigma_y[:, 1:, :])
    #     sigma_y_Ey = sigma_y_Ey[1:-1, :, 1:-1]
    #
    #     epsilon_Ey = 0.5 * (epsilon[:, :-1, :] + epsilon[:, 1:, :])
    #     epsilon_Ey = epsilon_Ey[1:-1, :, 1:-1]
    #
    #     Ey[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Ey) / (1 + dt / 2 * sigma_x_Ey)) * Ey[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            1 / (epsilon_Ey * (1 + dt / 2 * sigma_x_Ey)) * \
    #                            ((1 + dt / 2 * sigma_y_Ey) * Dy[1:-1, 1:-1, 1:-1] - (
    #                                    1 - dt / 2 * sigma_y_Ey) * Dy_old[1:-1, 1:-1, 1:-1])
    #
    #     sigma_y_Ez = 0.5 * (sigma_y[:, :, :-1] + sigma_y[:, :, 1:])
    #     sigma_y_Ez = sigma_y_Ez[1:-1, 1:-1, :]
    #
    #     sigma_z_Ez = 0.5 * (sigma_z[:, :, :-1] + sigma_z[:, :, 1:])
    #     sigma_z_Ez = sigma_z_Ez[1:-1, 1:-1, :]
    #
    #     epsilon_Ez = 0.5 * (epsilon[:, :, :-1] + epsilon[:, :, 1:])
    #     epsilon_Ez = epsilon_Ez[1:-1, 1:-1, :]
    #
    #     Ez[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Ez) / (1 + dt / 2 * sigma_y_Ez)) * Ez[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            1 / (epsilon_Ez * (1 + dt / 2 * sigma_y_Ez)) * \
    #                            ((1 + dt / 2 * sigma_z_Ez) * Dz[1:-1, 1:-1, 1:-1] - (
    #                                    1 - dt / 2 * sigma_z_Ez) * Dz_old[1:-1, 1:-1, 1:-1])
    #
    #     sigma_y_Bx = 0.5 * (sigma_y[:, :-1, :] + sigma_y[:, 1:, :])
    #     sigma_y_Bx = 0.5 * (sigma_y_Bx[:, :, :-1] + sigma_y_Bx[:, :, 1:])
    #     sigma_y_Bx = sigma_y_Bx[1:-1, :, :]
    #
    #     Bx[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_y_Bx) / (1 + dt / 2 * sigma_y_Bx)) * Bx[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_y_Bx) * (
    #                                    (Ey[1:-1, 1:-1, 1:] - Ey[1:-1, 1:-1, :-1]) / dz -
    #                                    (Ez[1:-1, 1:, 1:-1] - Ez[1:-1, :-1, 1:-1]) / dy
    #                            )
    #
    #     sigma_z_By = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    #     sigma_z_By = 0.5 * (sigma_z_By[:, :, :-1] + sigma_z_By[:, :, 1:])
    #     sigma_z_By = sigma_z_By[:, 1:-1, :]
    #
    #     By[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_z_By) / (1 + dt / 2 * sigma_z_By)) * By[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_z_By) * (
    #                                    (Ez[1:, 1:-1, 1:-1] - Ez[:-1, 1:-1, 1:-1]) / dx -
    #                                    (Ex[1:-1, 1:-1, 1:] - Ex[1:-1, 1:-1, :-1]) / dz
    #                            )
    #
    #     sigma_x_Bz = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    #     sigma_x_Bz = 0.5 * (sigma_x_Bz[:, :-1, :] + sigma_x_Bz[:, 1:, :])
    #     sigma_x_Bz = sigma_x_Bz[:, :, 1:-1]
    #     Bz[1:-1, 1:-1, 1:-1] = ((1 - dt / 2 * sigma_x_Bz) / (1 + dt / 2 * sigma_x_Bz)) * Bz[1:-1,
    #                                                                                      1:-1,
    #                                                                                      1:-1] + \
    #                            dt / (1 + dt / 2 * sigma_x_Bz) * (
    #                                    (Ex[1:-1, 1:, 1:-1] - Ex[1:-1, :-1, 1:-1]) / dy -
    #                                    (Ey[1:, 1:-1, 1:-1] - Ey[:-1, 1:-1, 1:-1]) / dx
    #                            )
    #
    #     sigma_z_Hx = 0.5 * (sigma_z[:, :-1, :] + sigma_z[:, 1:, :])
    #     sigma_z_Hx = 0.5 * (
    #             sigma_z_Hx[:, :, :-1] + sigma_z_Hx[:, :, 1:])
    #
    #     sigma_z_Hx_in = sigma_z_Hx[1:-1, :, :]
    #
    #     sigma_x_Hx = 0.5 * (sigma_x[:, :-1, :] + sigma_x[:, 1:, :])
    #     sigma_x_Hx = 0.5 * (sigma_x_Hx[:, :, :-1] + sigma_x_Hx[:, :, 1:])
    #     sigma_x_Hx_in = sigma_x_Hx[1:-1, :, :]
    #
    #     mu_Hx = 0.5 * (mu[:, :-1, :] + mu[:, 1:, :])
    #     mu_Hx = 0.5 * (mu_Hx[:, :, :-1] + mu_Hx[:, :, 1:])
    #     mu_Hx_in = mu_Hx[1:-1, :, :]
    #
    #     Hx[1:-1, 1:-1, 1:-1] = (
    #             ((1 - dt / 2 * sigma_z_Hx_in) / (1 + dt / 2 * sigma_z_Hx_in)) * Hx[1:-1, 1:-1,
    #                                                                             1:-1] +
    #             1 / (mu_Hx_in * (1 + dt / 2 * sigma_z_Hx_in)) *
    #             (
    #                     (1 + dt / 2 * sigma_x_Hx_in) * Bx[1:-1, 1:-1, 1:-1] -
    #                     (1 - dt / 2 * sigma_x_Hx_in) * Bx_old[1:-1, 1:-1, 1:-1]
    #             )
    #     )
    #
    #     sigma_x_Hy = 0.5 * (sigma_x[:-1, :, :] + sigma_x[1:, :, :])
    #     sigma_x_Hy = 0.5 * (sigma_x_Hy[:, :, :-1] + sigma_x_Hy[:, :, 1:])
    #     sigma_x_Hy_in = sigma_x_Hy[:, 1:-1, :]
    #
    #     sigma_y_Hy = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])
    #     sigma_y_Hy = 0.5 * (sigma_y_Hy[:, :, :-1] + sigma_y_Hy[:, :, 1:])
    #     sigma_y_Hy_in = sigma_y_Hy[:, 1:-1, :]
    #
    #     mu_Hy = 0.5 * (mu[:-1, :, :] + mu[1:, :, :])
    #     mu_Hy = 0.5 * (mu_Hy[:, :, :-1] + mu_Hy[:, :, 1:])
    #     mu_Hy_in = mu_Hy[:, 1:-1, :]
    #     Hy[1:-1, 1:-1, 1:-1] = (
    #             ((1 - dt / 2 * sigma_x_Hy_in) / (1 + dt / 2 * sigma_x_Hy_in)) * Hy[1:-1, 1:-1,
    #                                                                             1:-1] +
    #             1 / (mu_Hy_in * (1 + dt / 2 * sigma_x_Hy_in)) *
    #             (
    #                     (1 + dt / 2 * sigma_y_Hy_in) * By[1:-1, 1:-1, 1:-1] -
    #                     (1 - dt / 2 * sigma_y_Hy_in) * By_old[1:-1, 1:-1, 1:-1]
    #             )
    #     )
    #
    #     sigma_y_Hz = 0.5 * (sigma_y[:-1, :, :] + sigma_y[1:, :, :])
    #     sigma_y_Hz = 0.5 * (sigma_y_Hz[:, :-1, :] + sigma_y_Hz[:, 1:, :])
    #     sigma_y_Hz_in = sigma_y_Hz[:, :, 1:-1]
    #
    #     sigma_z_Hz = 0.5 * (sigma_z[:-1, :, :] + sigma_z[1:, :, :])
    #     sigma_z_Hz = 0.5 * (sigma_z_Hz[:, :-1, :] + sigma_z_Hz[:, 1:, :])
    #     sigma_z_Hz_in = sigma_z_Hz[:, :, 1:-1]
    #
    #     mu_Hz = 0.5 * (mu[:-1, :, :] + mu[1:, :, :])
    #     mu_Hz = 0.5 * (mu_Hz[:, :-1, :] + mu_Hz[:, 1:, :])
    #     mu_Hz_in = mu_Hz[:, :, 1:-1]
    #
    #     Hz[1:-1, 1:-1, 1:-1] = (
    #             ((1 - dt / 2 * sigma_y_Hz_in) / (1 + dt / 2 * sigma_y_Hz_in)) * Hz[1:-1, 1:-1,
    #                                                                             1:-1] +
    #             1 / (mu_Hz_in * (1 + dt / 2 * sigma_y_Hz_in)) *
    #             (
    #                     (1 + dt / 2 * sigma_z_Hz_in) * Bz[1:-1, 1:-1, 1:-1] -
    #                     (1 - dt / 2 * sigma_z_Hz_in) * Bz_old[1:-1, 1:-1, 1:-1]
    #             )
    #     )
    #
    #     Dx_old = cp.copy(Dx)
    #     Dy_old = cp.copy(Dy)
    #     Dz_old = cp.copy(Dz)
    #     Bx_old = cp.copy(Bx)
    #     By_old = cp.copy(By)
    #     Bz_old = cp.copy(Bz)

        # return Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old
