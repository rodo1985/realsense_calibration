import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

class realsense:

    def __init__(self):
        """
        Constructor
        """
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def __del__(self):
        self.pipeline.stop()

    def wait_for_frame(self):
        """

        :return: aligned_depth_Frame and aligned_color_image
        """
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return None

        aligned_color_image = np.asanyarray(color_frame.get_data())

        return aligned_depth_frame, aligned_color_image

    def remove_background(self, aligned_depth_frame, color_image, near_fov=0, far_fov=20):
        """
        Method to remove the pixels out of near_fov and far_fov and not value pixels
        :param aligned_depth_frame:
        :param color_image:
        :param near_fov:
        :param far_fov:
        :return:
        """

        # Remove background - Set pixels further than clipping_distance to grey
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        grey_color = 0
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where(((depth_image_3d > near_fov / self.depth_scale) &
                               (depth_image_3d < far_fov / self.depth_scale)) |
                              (depth_image_3d <= 0), grey_color,
                              color_image)

        return bg_removed

    @staticmethod
    def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y):
        """
        Convert the depth and image point information to metric coordinates
        :param depth: The depth value of the image point
        :param pixel_x: The x value of the image coordinate
        :param pixel_y: The y value of the image coordinate
        :return: np array with x, y and z
        """

        camera_intrinsics = depth.profile.as_video_stream_profile().intrinsics

        Z = depth.get_distance(pixel_x, pixel_y)

        # Coordinates from the camera center, it could be different from realsense viewer
        X = (Z * (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx) - 0.0175
        Y = Z * (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy

        return np.array([X, Y, Z])

    @staticmethod
    def draw_coordinate_system(img, corners, imgpts):
        """

        :param img:
        :param corners:
        :param imgpts:
        :return:
        """
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        return img

    @staticmethod
    def cam_to_robot_base(camera_point, t_cam_flange, rot_cam_flange, t_base_flange, rot_base_flange):
        """

        :param camera_point: np array con la coordenada del punto encontrado por la camara
        :param t_cam_flange: translacion de la camara al flange extraida de la calibracion
        :param rot_cam_flange: scipy.spatial.transform rotacion entre camara y flange
        :param t_base_flange: np array translacion del flande a la base
        :param rot_base_flange: scipy.spatial.transform rotacion entre flange y base
        :return: devuelve el punto en base robot
        """

        return t_base_flange + rot_base_flange.apply(t_cam_flange +
                                                     rot_cam_flange.apply(camera_point, inverse=False),
                                                     inverse=False)

    @staticmethod
    def checkerboard_calibration(img, nx, ny):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        objp = objp

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d pionts in image plane.

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findCirclesGrid(gray, (nx, ny), None)
        rvecs = []
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)

            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners)

            # Get image size
            img_size = (img.shape[1], img.shape[0])

            # Do camera calibration given object points and image points
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

            axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, dist)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            # img = realsense.draw_coordinate_system(img, corners, imgpts)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        return ret, img, imgpoints, rvecs