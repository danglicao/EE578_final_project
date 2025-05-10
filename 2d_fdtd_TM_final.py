import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

def main():
    # Simulation parameters
    epsilon0 = 8.85e-12
    mu0 = 4* np.pi * 1e-7
    c0 = 1/np.sqrt(epsilon0*mu0)
    lambda_0 = 300e-3
    lambda_U = lambda_0 * 1.6
    lambda_L = lambda_0 * 0.7
    dx = dy = 5e-3
    dt = 0.99* dx / (c0 * np.sqrt(2))

    # PML Setup
    pml_thickness = 16
    sigma_max = 1e11

    x_min, x_max = -200e-3, 200e-3
    y_min, y_max = -200e-3, 200e-3
    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1

    nt = 400
    dp_x, dp_y= 0, 0
    x_prob, y_prob = x_min+(pml_thickness+1)*dx, y_min+(pml_thickness+1)*dy
    i_x_src = int(round((dp_x - x_min) / dx))
    i_y_src = int(round((dp_y - y_min) / dy))
    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_dp_x = int(round((dp_x - x_min) / dx))
    i_dp_y = int(round((dp_y - y_min) / dy))

    dp_len = int(round(lambda_0/2/dy))
    i_dp_x_min = i_dp_x - dp_len
    i_dp_x_max = i_dp_x + dp_len + 1
    Conductor = np.ones((Nx, Ny), dtype = np.float32)
    Conductor[i_dp_x_min:i_dp_x_max,i_dp_y] = 0
    Conductor[i_dp_x,i_dp_y] = 1

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

    sigma_x, sigma_y = pml_profile(sigma_max, pml_thickness, Nx, Ny)
    Ez_record = np.zeros(int(nt), dtype = np.float32)

    fig, ax = plt.subplots(figsize=(6,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    extent = [x_min, x_max, y_min, y_max]

    im = ax.imshow(Ez, extent=extent, vmin=-0.8e-25, vmax=0.8e-25, cmap='RdBu')
    cb = fig.colorbar(im, cax=cax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Ez field evolution")

    frames = []

    for n in range(nt):
        Ez[i_x_src][i_y_src] += dt * epsilon[i_x_src][i_y_src] * gaussian_source(n, dt, sigma, omega_0)

        Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old = update_equations(Dz, Ez, Hx, Hy, Bx, By,
            Dz_old, Bx_old, By_old,
            sigma_x, sigma_y, epsilon, mu,
            dt, dx, dy, Conductor)

        Ez_record[n] = Ez[i_x_prob][i_y_prob]

        if n % 2 == 0:
            frame_list = []
            img = ax.imshow(Ez.copy(), extent=extent, vmin=-0.8e-25, vmax=0.8e-25, cmap='RdBu', animated=True)
            frame_list.append(img)
            # Draw the red box to indicate PML location
            rect = patches.Rectangle((x_min+pml_thickness*dx, y_min+pml_thickness*dy),
                                     x_max-x_min-2*dx*pml_thickness, y_max-y_min-2*dy*pml_thickness,
                                     fc='none', ec='r', lw=2, animated=True)
            ax.add_patch(rect)
            frame_list.append(rect)
            frames.append(frame_list)

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
    ani.save('Ez_field_with_colorbar.gif', writer='pillow')
    plt.show()

    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(np.arange(nt), Ez_record)
    ax.set_xlabel('Time (5 ms)')
    ax.set_ylabel('Ez at probe point')
    ax.set_title('Field at Probe Point')
    plt.show()

def gaussian_source(n, dt, sigma, omega0):
    t_now = (n - 0.5) * dt
    t0 = 4 * sigma
    return np.exp(-((t_now - t0) / sigma)**2) * np.sin(omega0 * (t_now - t0))

def sigma_profile(sigma_max, pml_thickness, distance):
    poly_deg = 3
    return sigma_max * (distance/pml_thickness)**poly_deg

def pml_profile(sigma_max, pml_thickness, Nx, Ny):
    sigma_x = np.zeros((Nx, Ny))
    sigma_y = np.zeros((Nx, Ny))
    for i in range(pml_thickness):
        sigma_x[i, :] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
        sigma_x[-1 - i, :] = sigma_profile(sigma_max, pml_thickness, pml_thickness - i)
    for j in range(pml_thickness):
        sigma_y[:, j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
        sigma_y[:, -1 - j] = sigma_profile(sigma_max, pml_thickness, pml_thickness - j)
    return sigma_x, sigma_y

def update_equations(Dz, Ez, Hx, Hy, Bx, By,
                     Dz_old, Bx_old, By_old,
                     sigma_x, sigma_y, epsilon, mu,
                     dt, dx, dy,
                     Conductor):
    # Dz, Ez
    Dz[1:-1, 1:-1] = ((1 - dt/2 * sigma_x[1:-1, 1:-1]) / (1 + dt/2 * sigma_x[1:-1, 1:-1])) * Dz[1:-1, 1:-1] + \
        dt / (1 + dt/2 * sigma_x[1:-1, 1:-1]) / epsilon[1:-1, 1:-1] * \
        ((Hy[1:, 1:-1] - Hy[:-1,1:-1]) / dx - (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy)
    Ez[1:-1, 1:-1] = ((1 - dt/2 * sigma_y[1:-1, 1:-1]) / (1 + dt/2 * sigma_y[1:-1, 1:-1])) * Ez[1:-1, 1:-1] + \
        1 / (1 + dt/2 * sigma_y[1:-1, 1:-1]) * (Dz[1:-1, 1:-1] - Dz_old[1:-1, 1:-1])
    Ez[1:-1, 1:-1] = Ez[1:-1, 1:-1]*Conductor[1:-1, 1:-1]

    # Bx, Hx
    sigma_y_Bx = 0.5 * (sigma_y[:, :-1] + sigma_y[:, 1:])
    sigma_x_Hx = 0.5 * (sigma_x[:, :-1] + sigma_x[:, 1:])
    mu_Hx = 0.5 * (mu[:, :-1] + mu[:, 1:])
    Bx[:-1, :] = ((1 - dt/2 * sigma_y_Bx[:-1, :]) / (1 + dt/2 * sigma_y_Bx[:-1, :])) * Bx[:-1, :] - \
        dt / (1 + dt/2 * sigma_y_Bx[:-1, :]) / mu_Hx[:-1, :] * (Ez[:-1, 1:] - Ez[:-1, :-1]) / dy
    Hx[:-1, :] = Hx[:-1, :] + \
            (1 + dt/2 * sigma_x_Hx[:-1, :]) * Bx[:-1,:] - \
            (1 - dt/2 * sigma_x_Hx[:-1, :]) * Bx_old[:-1, :]

    # By, Hy
    sigma_x_By = 0.5 * (sigma_x[:-1, :] + sigma_x[1:, :])
    sigma_y_Hy = 0.5 * (sigma_y[:-1, :] + sigma_y[1:, :])
    mu_Hy = 0.5 * (mu[:-1, :] + mu[1:, :])
    By[:, :-1] = ((1 - dt/2 * sigma_x_By[:, :-1]) / (1 + dt/2 * sigma_x_By[:, :-1])) * By[:, :-1] + \
        dt / (1 + dt/2 * sigma_x_By[:, :-1]) / mu_Hy[:, :-1] * (Ez[1:, :-1] - Ez[:-1, :-1]) / dx
    Hy[:, :-1] = Hy[:, :-1] + (1 + dt / 2 * sigma_y_Hy[:, :-1]) * By[:, :-1] - \
        (1 - dt/2 * sigma_y_Hy[:, :-1]) * By_old[:, :-1]

    Dz_old = np.copy(Dz)
    Bx_old = np.copy(Bx)
    By_old = np.copy(By)

    return Dz, Ez, Hx, Hy, Bx, By, Dz_old, Bx_old, By_old

if __name__ == "__main__":
    main()