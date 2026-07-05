import numpy as np
import math

# Intrinsics (pixels)
fx = 567.720776478944
fy = 577.2136917114258
px = 628.72078
py = 369.30687

# Image size (after any crop, set accordingly)
W, H = 1280, 640  # update if needed

def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa,  ca, 0],
                     [ 0,   0, 1]], dtype=float)

def pixel_shift(u, v, yaw, pitch, roll):
    # Build rotation (change order here if your pipeline differs)
    R = Rz(roll) @ Ry(yaw) @ Rx(pitch)

    x = (u - px) / fx
    y = (v - py) / fy
    r = np.array([x, y, 1.0], dtype=float)

    rp = R @ r
    up = fx * (rp[0] / rp[2]) + px
    vp = fy * (rp[1] / rp[2]) + py

    return up - u, vp - v

# Example: +5 deg applied to yaw, pitch, roll (worst-case deterministic)
deg = 0.5
yaw = math.radians(deg)
pitch = math.radians(deg)
roll = math.radians(deg)

# Report corner shifts
corners = [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]
print("Corner pixel shifts (du, dv) in pixels:")
for (u, v) in corners:
    du, dv = pixel_shift(u, v, yaw, pitch, roll)
    print(f"  (u,v)=({u:4d},{v:4d}) -> du={du:8.2f}, dv={dv:8.2f}, |d|={math.hypot(du,dv):8.2f}")

# Max shift over a coarse grid (adjust step for speed/precision)
step = 20
max_mag = 0.0
max_loc = None
for v in range(0, H, step):
    for u in range(0, W, step):
        du, dv = pixel_shift(u, v, yaw, pitch, roll)
        mag = math.hypot(du, dv)
        if mag > max_mag:
            max_mag = mag
            max_loc = (u, v, du, dv)

u, v, du, dv = max_loc
print(f"\nMax |shift| over grid(step={step}): {max_mag:.2f} px at (u,v)=({u},{v}) with du={du:.2f}, dv={dv:.2f}")
