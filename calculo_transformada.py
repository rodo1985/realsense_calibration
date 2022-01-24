from scipy.spatial.transform import Rotation as R
from modules.realsense import realsense
import numpy as np

# init realsense
# rs = realsense()

point = np.array([48, 267, 1062])

# point = np.array([-6.001, -3.778, 470.712])

# calibration
t_cam_flange = np.array([5.68945, -18.2781, 178.771])
rot_cam_flange = R.from_matrix([
    [-0.0730971, -0.997282,	-0.00926262],
    [0.997322, -0.073072, -0.00302439],
    [0.00233933, -0.00945889, 0.999953]])
# [ x: 0.165205, y: -0.5472205, z: 91.6236921 ]

# # robot pose
# t_base_flange = np.array([196.53,-1534.6, 1156.53])
# rot_base_flange = R.from_euler('zyx', [-30.68, 0.61, 178.91], degrees=True)

# robot pose
t_base_flange = np.array([507.34, 1223.44, 1545.22])

rot_base_flange = R.from_euler('zyx', [-32.46, 0, -180], degrees=True)


object_to_base = t_base_flange + rot_base_flange.apply(t_cam_flange +
                                                       rot_cam_flange.apply(point, inverse=True),
                                                       inverse=False)

print(object_to_base)