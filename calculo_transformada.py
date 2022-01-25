from scipy.spatial.transform import Rotation as R
from modules.realsense import realsense
import numpy as np
import open3d as o3d
import copy

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


# intento de visualizar fail
# # base robot
# robot_base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
# mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5,
#                                                 height=0.5,
#                                                 depth=0.5)

# # flange
# robot_flange_frame = copy.deepcopy(robot_base_frame)
# R = robot_base_frame.get_rotation_matrix_from_xyz(rot_base_flange.as_euler('xyz', degrees=False))
# robot_flange_frame.rotate(R, center=(0, 0, 0))
# robot_flange_frame.translate(t_base_flange/100)

# #camara
# camera_frame = copy.deepcopy(robot_flange_frame)
# R = robot_base_frame.get_rotation_matrix_from_xyz(rot_cam_flange.as_euler('xyz', degrees=False))
# camera_frame.rotate(R, center=(0, 0, 0))
# camera_frame.translate(t_cam_flange/100)

# # objeto
# object_frame = copy.deepcopy(camera_frame).translate(point/100 *-1)

# # visualizar
# o3d.visualization.draw_geometries([robot_base_frame, robot_flange_frame, camera_frame, object_frame, mesh_box])

