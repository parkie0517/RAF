import math

# Camera intrinsics (pixels)
fx = 567.720776478944
fy = 577.2136917114258

# Image size after top crop
img_size_w = 1280
img_size_h = 640   # 720 -> 640 (crop 80 px from top)

# Field of View calculations
hfov_rad = 2.0 * math.atan(img_size_w / (2.0 * fx))
vfov_rad = 2.0 * math.atan(img_size_h / (2.0 * fy))

hfov_deg = math.degrees(hfov_rad)
vfov_deg = math.degrees(vfov_rad)

print(f"Horizontal FOV: {hfov_deg:.2f} degrees")
print(f"Vertical FOV:   {vfov_deg:.2f} degrees")
