import numpy as np
import cv2
from skspatial.objects import Plane
import math
from scipy.spatial.transform import Rotation as R
from modules.realsense import realsense


def main():

    # variables
    nx = 11
    ny = 8
    first_image = True

    # init realsense
    rs = realsense()

    # Streaming loop
    while True:
        
        # get frame
        aligned_depth_frame, aligned_color_image = rs.wait_for_frame()

        if not first_image:

            # get calibration
            ret, img, points, rvects = rs.checkerboard_calibration(aligned_color_image, nx, ny)
            coord = []
            if ret:
                for i in range(nx * ny):

                    x, y, z = rs.convert_depth_pixel_to_metric_coordinate(aligned_depth_frame, points[0][i][0][0],
                                                                       points[0][i][0][1])
                    point = np.array((x, y, z))
                    if point[0] != 0 and point[1] != 0 and point[2] != 0:
                        coord.append(point)

                plane = Plane.best_fit(np.array(coord))

                camera_point = np.array([point[0], point[1], point[2]])

                # calibration
                t_cam_flange = np.array([5.68945, -18.2781, 178.771])
                rot_cam_flange = R.from_matrix([
                    [-0.0730971, -0.997282, -0.00926262],
                    [0.997322, -0.073072, -0.00302439],
                    [0.00233933, -0.00945889, 0.999953]])

                # robot pose
                t_base_flange = np.array([196.53, -1534.6, 1156.53])
                rot_base_flange = R.from_euler('zyx', [-30.68, 0.61, 178.91], degrees=True)

                obect_to_base = rs.cam_to_robot_base(camera_point,
                                                     t_cam_flange,
                                                     rot_cam_flange,
                                                     t_base_flange,
                                                     rot_base_flange)
                print(camera_point)



            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', img)
            key = cv2.waitKey()

            # # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:

                cv2.destroyAllWindows()
                break
        else:
            first_image = False


if __name__ == '__main__':
    main()
