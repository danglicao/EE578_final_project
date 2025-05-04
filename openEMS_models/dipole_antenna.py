import os, tempfile
from pylab import *

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
import numpy as np
import matplotlib.pyplot as plt

Sim_Path = os.path.join(tempfile.gettempdir(), 'Dipole_1GHz')
post_proc_only = False


dipole_len      = 150
gap_len         = 5
wire_radius     = 5


SimBox = array([400, 400, 400])

dx = dy = dz = 5
Nx = Ny = Nz = 80


f0 = 1.0e9
fc = 1.42e9
NrTS = 1800


FDTD = openEMS(NrTS=NrTS, EndCriteria=1e-4)
# FDTD = openEMS(EndCriteria=1e-7)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8']*6)


CSX  = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
# mesh_res = C0/(f0+fc)/1e-3/20

mesh.AddLine('x', [-SimBox[0]/2,  SimBox[0]/2])
mesh.AddLine('y', [-SimBox[1]/2,  SimBox[1]/2])
mesh.AddLine('z', [-SimBox[2]/2,  SimBox[2]/2])


mesh.SmoothMeshLines('all', dx, 1.01)

dipole = CSX.AddMetal('dipole')



z_low  = -dipole_len/2.0          # -75 mm
z_high =  dipole_len/2.0          # +75 mm
z_gap  =  gap_len/2.0             #  ±2.5 mm
#

start_low = [0, 0, z_low ]
stop_low  = [0, 0, -z_gap]
dipole.AddCylinder(start=start_low, stop=stop_low, radius=wire_radius, priority=10)


start_high = [0, 0, +z_gap]
stop_high  = [0, 0, z_high]
dipole.AddCylinder(start=start_high, stop=stop_high, radius=wire_radius, priority=10)


port_start = [0, 0, -z_gap]
port_stop  = [0, 0, +z_gap]
feed_R     = 73          # [Ω]
port = FDTD.AddLumpedPort(1, feed_R, port_start, port_stop,
                          'z', 1.0, priority=5, edges2grid='xyz')

nf2ff = FDTD.CreateNF2FFBox()


if 0:
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX_file = os.path.join(Sim_Path, 'dipole.xml')
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))


if not post_proc_only:
    FDTD.Run(Sim_Path, verbose=3, cleanup=True)


# f = linspace(max(0.5e9, f0-fc), f0+fc, 601)
# f = linspace(0, 1.6e9, 1800)  # [Hz] 频率范围
dt =0.99 * 5e-3 / (3e8 * np.sqrt(np.float64(3)))
f = np.fft.fftfreq(1800, d = dt.item())
port.CalcPort(Sim_Path, f)


s11_dB = 20*log10(abs(port.uf_ref/port.uf_inc))
figure(); plot(f/1e9, s11_dB, 'k-', lw=2); grid()
xlabel('Frequency (GHz)'); ylabel('|S11| (dB)')
title('Reflection of 1 GHz Half-wave Dipole')


Zin = port.uf_tot/port.if_tot
figure()
plot(f/1e9, real(Zin), 'k-', lw=2, label='Re{Zin}')
plot(f/1e9, imag(Zin), 'r--', lw=2, label='Im{Zin}')
grid(); legend(); xlabel('Frequency (GHz)'); ylabel('Ω')
title('Input Impedance')
print(type(Zin))
print(os.getcwd())



idx = argmin(s11_dB)
f_res = f[idx]
if s11_dB[idx] < -10:
    theta = arange(-180, 180, 2)
    phi   = [0, 90]
    nf2 = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi, center=[0,0,0])
    figure()
    Enorm = 20 * np.log10(nf2.E_norm[0] / np.max(nf2.E_norm[0])) + nf2.Dmax[0]
    # Enorm = 20*log10(nf2.E_norm[0]/max(nf2.E_norm[0])) + nf2.Dmax[0]
    plot(theta, squeeze(Enorm[:,0]), 'k-', lw=2, label='xz-plane')
    plot(theta, squeeze(Enorm[:,1]), 'r--', lw=2, label='yz-plane')
    grid(); legend()
    xlabel('Theta (deg)'); ylabel('Directivity (dBi)')
    title('Radiation Pattern @ {:.3f} GHz'.format(f_res/1e9))
else:
    print('No far field pattern available, S11 < -10 dB at resonance frequency!')

show()

np.savetxt('your save path(need to be absolute or will be difficult to find)',Zin)
