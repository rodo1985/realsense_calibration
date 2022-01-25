from scipy.spatial.transform import Rotation as R
from modules.realsense import realsense
import numpy as np


point = np.array([111, 45, 318])


# calibration
t_cam_flange = np.array([13.0373, 72.0783, 37.6219])
rot_cam_flange = R.from_matrix([
    [-0.999883, 0.00485597,	0.0145292],
    [-0.00642179, -0.993939, -0.109744],
    [0.0139082, -0.109825, 0.993854]])

# robot pose
t_base_flange = np.array([-204.04, -294.36, 352.13,])
rot_base_flange = R.from_euler('xyz', [ -179.98, -0.01, -90.42], degrees=True)


object_to_base = t_base_flange + rot_base_flange.apply(t_cam_flange +
                                                       rot_cam_flange.apply(point, inverse=True),
                                                       inverse=False)

print(object_to_base)