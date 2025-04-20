import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tqdm import tqdm
from physics_functions import fdtd_functions
import matplotlib.animation as animation


def main():
    # Normalized
    # epsilon0 = cp.float32(1.0)
    # mu0 = cp.float32(1.0)
    # c0 = cp.float32(1.0)
    # sigma_max = 200

    # Not Normalized
    epsilon0 = cp.float32(8.85e-12)
    mu0 = cp.float32(4 * cp.pi * 1e-7)
    c0 = 1 / cp.sqrt(epsilon0 * mu0)
    # sigma_max = cp.float32(0.9e12) # without fixing J
    sigma_max = cp.float32(5e10)  # fixed J


    pml_thickness = 16
    # sigma_max  = -(cp.float32(3 + 1) / cp.float32(4)) * (c0 / cp.float32(pml_thickness)) * cp.log(cp.float32(1e-100))

    lambda_0 = cp.float32(100e-3)
    lambda_U = cp.float32(150e-3)
    lambda_L = cp.float32(50e-3)

    print(c0 / lambda_0 / 1e9)

    dx = dy = dz = cp.float32(5e-3)  # m, spatial step size
    dt = 0.99 * dx / (c0 * cp.sqrt(cp.float32(3)))

    x_min, x_max = -250e-3, 250e-3
    y_min, y_max = -250e-3, 250e-3
    z_min, z_max = -250e-3, 250e-3

    Nx = int(round((x_max - x_min) / dx)) + 1
    Ny = int(round((y_max - y_min) / dy)) + 1
    Nz = int(round((z_max - z_min) / dz)) + 1

    nt = int(500)

    x_src, y_src, z_src = 0, 0, 0
    tmp = 5
    # x_prob, y_prob, z_prob = 0 + tmp * dx, 0 + tmp * dy, 0 + tmp * dz
    x_prob, y_prob, z_prob = x_min + (pml_thickness + tmp) * dx, y_min + (
                pml_thickness + tmp) * dy, z_min + (pml_thickness + tmp) * dz

    i_x_src = int(round((x_src - x_min) / dx))
    i_y_src = int(round((y_src - y_min) / dy))
    i_z_src = int(round((z_src - z_min) / dz))

    i_x_prob = int(round((x_prob - x_min) / dx))
    i_y_prob = int(round((y_prob - y_min) / dy))
    i_z_prob = int(round((z_prob - z_min) / dz))

    Ex_record = cp.zeros(nt, dtype = cp.float32)
    Ey_record = cp.zeros(nt, dtype = cp.float32)
    Ez_record = cp.zeros(nt, dtype = cp.float32)
    Hx_record = cp.zeros(nt, dtype = cp.float32)
    Hy_record = cp.zeros(nt, dtype = cp.float32)
    Hz_record = cp.zeros(nt, dtype = cp.float32)

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

    sigma_x_vec, sigma_y_vec, sigma_z_vec = fdtd_functions.pml_profile(sigma_max, pml_thickness, Nx,
                                                                       Ny, Nz)
    sigma_x_3d, sigma_y_3d, sigma_z_3d = cp.meshgrid(sigma_x_vec, sigma_y_vec, sigma_z_vec,
                                                     indexing = 'ij')

    Ex_time_record = cp.zeros((Nx + 1, Ny, Nz), dtype = cp.float32)
    Ey_time_record = cp.zeros((Nx, Ny + 1, Nz), dtype = cp.float32)
    Ez_time_record = cp.zeros((Nx, Ny, Nz + 1), dtype = cp.float32)
    Hx_time_record = cp.zeros((Nx, Ny + 1, Nz + 1), dtype = cp.float32)
    Hy_time_record = cp.zeros((Nx + 1, Ny, Nz + 1), dtype = cp.float32)
    Hz_time_record = cp.zeros((Nx + 1, Ny + 1, Nz), dtype = cp.float32)

    params = fdtd_functions.fdtd_param_alignment(Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
                                                 sigma_x_3d, sigma_y_3d, sigma_z_3d, epsilon, mu)

    # Antenna hyper parameters
    L = cp.float32(lambda_0 / (2))
    L_rel = L // dz
    ic, jc, kc = Nx // 2, Ny // 2, Nz // 2

    half_len = L / 2
    half_len_rel = int(L_rel // 2)

    i_z_dipole_start = kc - half_len_rel
    i_z_dipole_end = kc + half_len_rel + 1

    # for animation
    save_every = 1
    n_frames = nt // save_every
    frame_idx = 0
    Ex_frames = cp.zeros((n_frames, Ny, Nz), dtype = cp.float32)
    Ey_frames = cp.zeros((n_frames, Nx, Nz), dtype = cp.float32)
    Ez_frames = cp.zeros((n_frames, Nx, Ny), dtype = cp.float32)


    sigma_x_3d[ic, jc, i_z_dipole_start:kc] = 1e8
    sigma_y_3d[ic, jc, i_z_dipole_start:kc] = 1e8
    sigma_z_3d[ic, jc, i_z_dipole_start:kc] = 1e8
    sigma_x_3d[ic, jc, kc + 1:i_z_dipole_end] = 1e8
    sigma_y_3d[ic, jc, kc + 1:i_z_dipole_end] = 1e8
    sigma_z_3d[ic, jc, kc + 1:i_z_dipole_end] = 1e8

    # main loop
    for n in tqdm(range(nt)):
        # add source
        Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz, Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old = fdtd_functions.update_equations(
            Dx, Dy, Dz, Ex, Ey, Ez, Hx, Hy, Hz, Bx, By, Bz,
            Dx_old, Dy_old, Dz_old, Bx_old, By_old, Bz_old,
            params,
            dt, dx, dy, dz)
        Ez[i_x_src][i_y_src][i_z_src] += dt * fdtd_functions.gaussian_source(n, dt, sigma, omega_0)

        Ex_record[n] = Ex[i_x_prob, i_y_prob, i_z_prob]
        Ey_record[n] = Ey[i_x_prob, i_y_prob, i_z_prob]
        Ez_record[n] = Ez[i_x_prob, i_y_prob, i_z_prob]
        Hx_record[n] = Hx[i_x_prob, i_y_prob, i_z_prob]
        Hy_record[n] = Hy[i_x_prob, i_y_prob, i_z_prob]
        Hz_record[n] = Hz[i_x_prob, i_y_prob, i_z_prob]

        if n % save_every == 0:
            Ex_frames[frame_idx] = Ex[ic, :, :]
            Ey_frames[frame_idx] = Ey[:, jc, :]
            Ez_frames[frame_idx] = Ez[:, :, kc]
            frame_idx += 1

        if n == 80:
            Ex_time_record = cp.copy(Ex)
            Ey_time_record = cp.copy(Ey)
            Ez_time_record = cp.copy(Ez)
            Hx_time_record = cp.copy(Hx)
            Hy_time_record = cp.copy(Hy)
            Hz_time_record = cp.copy(Hz)

    Ex_time_record_cpu = Ex_time_record.get()
    Ey_time_record_cpu = Ey_time_record.get()
    Ez_time_record_cpu = Ez_time_record.get()
    Hx_time_record_cpu = Hx_time_record.get()
    Hy_time_record_cpu = Hy_time_record.get()
    Hz_time_record_cpu = Hz_time_record.get()

    Ex_frames_cpu = Ex_frames.get()
    Ey_frames_cpu = Ey_frames.get()
    Ez_frames_cpu = Ez_frames.get()

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
    t = np.arange(nt) * dt.get()
    # print(Ex_record_cpu.shape)
    # print(Ex_record_cpu)
    plot_probe_fields(t, Ex_record_cpu, Ey_record_cpu, Ez_record_cpu, Hx_record_cpu, Hy_record_cpu,
                      Hz_record_cpu)
    generate_field_animation(Ex_frames_cpu, save_every = save_every,save_path = "figs/Ex_animation.mp4")
    generate_field_animation(Ey_frames_cpu, save_every = save_every,save_path = "figs/Ey_animation.mp4")
    generate_field_animation(Ez_frames_cpu, save_every = save_every,save_path = "figs/Ez_animation.mp4")

    # Freespace Impedance
    E = np.sqrt(
        Ex_record_cpu ** 2 + Ey_record_cpu ** 2 + Ez_record_cpu ** 2)  # np.sqrt(Ex_record_cpu**2 + Ey_record_cpu**2 + Ez_record_cpu**2)
    H = np.sqrt(
        Hx_record_cpu ** 2 + Hy_record_cpu ** 2 + Hz_record_cpu ** 2)  # np.sqrt(Hx_record_cpu**2 + Hy_record_cpu**2 + Hz_record_cpu**2)
    E_f = np.fft.fft(E)
    H_f = np.fft.fft(H)

    Ex_f = np.fft.fft(Ex_record_cpu)
    Ey_f = np.fft.fft(Ey_record_cpu)
    Hx_f = np.fft.fft(Hx_record_cpu)
    Hy_f = np.fft.fft(Hy_record_cpu)

    freqs_hz = np.fft.fftfreq(nt, d = dt.item())
    half = nt // 2
    freqs_hz = freqs_hz[:half]
    plt.figure(figsize = (6, 4))

    E_f = E_f[:half]
    H_f = H_f[:half]
    Z = np.divide(abs(E_f), abs(H_f), out = np.zeros_like(abs(E_f)), where = (abs(H_f) != 0))
    # print(Z)

    Ex_f = Ex_f[:half]
    Ey_f = Ey_f[:half]
    Hx_f = Hx_f[:half]
    Hy_f = Hy_f[:half]
    Z_fs1 = np.divide(abs(Ex_f), abs(Hy_f), out = np.zeros_like(abs(Ex_f)),
                      where = (abs(Hy_f) != 0))
    Z_fs2 = np.divide(abs(Ey_f), abs(Hx_f), out = np.zeros_like(abs(Ey_f)),
                      where = (abs(Hx_f) != 0))
    plt.plot(freqs_hz / 1e9, Z_fs1, label = "Ex/Hy")
    plt.plot(freqs_hz / 1e9, Z_fs2, label = "Ey/Hx", linewidth = 3, linestyle = 'dotted')
    plt.plot(freqs_hz / 1e9, Z, label = "|E|/|H|")
    plt.vlines(2.998, -2, 1e4, colors = "k")
    plt.legend()
    plt.grid(True)
    plt.xlim(c0 / lambda_U / 1e9, c0 / lambda_L / 1e9)
    plt.ylim(-2, 500)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Impedance")
    plt.title("Freespace Impedance")
    plt.tight_layout()
    plt.show()
    print(np.sqrt(mu0 / epsilon0))

def plot_final_fields(Ex, Ey, Ez, Hx, Hy, Hz, Nx, Ny, Nz):
    kx = Nx // 2
    ky = Ny // 2
    kz = Nz // 2
    plt.figure(figsize = (15, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(Ex[kx, :, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Ex at z = center')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(Ey[:, ky, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Ey at z = center')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(Ez[:, :, kz], cmap = 'RdBu', origin = 'lower')
    plt.title('Ez at z = center')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(Hx[kx, :, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Hx at z = center')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(Hy[:, ky, :], cmap = 'RdBu', origin = 'lower')
    plt.title('Hy at z = center')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.imshow(Hz[:, :, kz], cmap = 'RdBu', origin = 'lower')
    plt.title('Hz at z = center')
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

def generate_field_animation(
        field_frames,
        save_path = "figs/field_animation.mp4",
        save_every = 1,
        cmap = "RdBu",
        vmin = None,
        vmax = None,
        fps = 25,
        dpi = 200
):

    n_frames = field_frames.shape[0]

    if vmin is None:
        # vmin = field_frames.min()
        vmin = np.percentile(field_frames, 2)
    if vmax is None:
        # vmax = field_frames.max()
        vmax = np.percentile(field_frames, 98)

    fig, ax = plt.subplots()
    im = ax.imshow(
        field_frames[0],
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
    print(f"✅ saved animation as：{save_path}")



if __name__ == "__main__":
    main()