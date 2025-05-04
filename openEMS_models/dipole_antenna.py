# -*- coding: utf-8 -*-
"""
 Simple Half-wave Dipole (1 GHz) -- openEMS Python 示例
 模板风格源自 “Simple Patch Antenna Tutorial”

 Tested with
  - python 3.10
  - openEMS v0.0.35

 (c) 2025  Dangli Cao (adapted from Thorsten Liebig)

"""

# === 1. 载入库 ===============================================================
import os, tempfile
from pylab import *

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *
import numpy as np
import matplotlib.pyplot as plt

# === 2. 全局参数 =============================================================
Sim_Path = os.path.join(tempfile.gettempdir(), 'Dipole_1GHz')
post_proc_only = False

# ── 几何尺寸（全部用毫米方便与示例一致） ──────────────────────────
dipole_len      = 150      # [mm] 全长
gap_len         = 5        # [mm] 馈电缝隙
wire_radius     = 5        # [mm] 导线半径

# 模拟域 400 mm × 400 mm × 400 mm（与 #domain 提示一致）
SimBox = array([400, 400, 400])       # [mm]

# ── 网格参数 ───────────────────────────────────────────────────
dx = dy = dz = 5           # [mm] 均匀立方网格
Nx = Ny = Nz = 80          # => 80³

# ── FDTD 与激励 ───────────────────────────────────────────────
f0 = 1.0e9                 # 中心频率 1 GHz
fc = 1.42e9                 # –20 dB 角频带宽
NrTS = 1800               # 时间步

# === 3. FDTD 设置 ============================================================
FDTD = openEMS(NrTS=NrTS, EndCriteria=1e-4)
# FDTD = openEMS(EndCriteria=1e-7)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(['PML_8']*6)

# === 4. 创建 CSXCAD 场景与网格 ===============================================
CSX  = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)           # 后续坐标全部以 mm 为单位
# mesh_res = C0/(f0+fc)/1e-3/20

# ── 初始化“空盒子”网格线（与示例相同做法） ────────────────────
mesh.AddLine('x', [-SimBox[0]/2,  SimBox[0]/2])
mesh.AddLine('y', [-SimBox[1]/2,  SimBox[1]/2])
mesh.AddLine('z', [-SimBox[2]/2,  SimBox[2]/2])

# 再均匀细化到 5 mm 间距
mesh.SmoothMeshLines('all', dx, 1.01)

# === 5. 建立几何：两段 PEC 圆柱 + Lumped Port ================================
dipole = CSX.AddMetal('dipole')
# dipole.Clear()   # 删掉原来的圆柱
# dipole.AddWire([0,0,-75], [0,0,-2.5], radius=0.7, max_seg_length=2)
# dipole.AddWire([0,0, 75], [0,0, 2.5], radius=0.7, max_seg_length=2)
# FDTD.AddEdges2Grid(dirs='xyz', properties=dipole, metal_edge_res=1)

# 计算 z 方向的两个端点（以 0 为中心）
z_low  = -dipole_len/2.0          # -75 mm
z_high =  dipole_len/2.0          # +75 mm
z_gap  =  gap_len/2.0             #  ±2.5 mm
#
# # 下半段 Cyl  (-75 mm → -2.5 mm)
start_low = [0, 0, z_low ]
stop_low  = [0, 0, -z_gap]
dipole.AddCylinder(start=start_low, stop=stop_low, radius=wire_radius, priority=10)

# 上半段 Cyl  (+2.5 mm → +75 mm)
start_high = [0, 0, +z_gap]
stop_high  = [0, 0, z_high]
dipole.AddCylinder(start=start_high, stop=stop_high, radius=wire_radius, priority=10)

# Lumped Port 跨越 5 mm 缝隙
port_start = [0, 0, -z_gap]
port_stop  = [0, 0, +z_gap]
feed_R     = 73          # [Ω]
port = FDTD.AddLumpedPort(1, feed_R, port_start, port_stop,
                          'z', 1.0, priority=5, edges2grid='xyz')

# === 6. 近-远场转换盒 (nf2ff) ================================================
nf2ff = FDTD.CreateNF2FFBox()

# === 7. 调试可视化 (可选) ====================================================
if 0:     # 调试时改成 1
    if not os.path.exists(Sim_Path):
        os.mkdir(Sim_Path)
    CSX_file = os.path.join(Sim_Path, 'dipole.xml')
    CSX.Write2XML(CSX_file)
    from CSXCAD import AppCSXCAD_BIN
    os.system(AppCSXCAD_BIN + ' "{}"'.format(CSX_file))

# === 8. 运行 FDTD ============================================================
if not post_proc_only:
    FDTD.Run(Sim_Path, verbose=3, cleanup=True)

# === 9. 后处理与绘图 =========================================================
# f = linspace(max(0.5e9, f0-fc), f0+fc, 601)
# f = linspace(0, 1.6e9, 1800)  # [Hz] 频率范围
dt =0.99 * 5e-3 / (3e8 * np.sqrt(np.float64(3)))
f = np.fft.fftfreq(1800, d = dt.item())
port.CalcPort(Sim_Path, f)

# S11
s11_dB = 20*log10(abs(port.uf_ref/port.uf_inc))
figure(); plot(f/1e9, s11_dB, 'k-', lw=2); grid()
xlabel('Frequency (GHz)'); ylabel('|S11| (dB)')
title('Reflection of 1 GHz Half-wave Dipole')

# 输入阻抗
Zin = port.uf_tot/port.if_tot
figure()
plot(f/1e9, real(Zin), 'k-', lw=2, label='Re{Zin}')
plot(f/1e9, imag(Zin), 'r--', lw=2, label='Im{Zin}')
grid(); legend(); xlabel('Frequency (GHz)'); ylabel('Ω')
title('Input Impedance')
print(type(Zin))
print(os.getcwd())
np.savetxt('C:/USC/EE578/final_project/data/openEms_Z.txt',Zin)

# 查找谐振点并做远场
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
    print('警告：未找到 <-10 dB 的共振点，未计算远场')

show()
