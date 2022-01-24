import math

import cv2
import numpy as np
from skspatial.objects import Plane


def find_aruco(image):
    """
    Find aruco in an image
    :param image:
    :return:
    aruco_id : id
    image : with a bounding box on the aruco
    corners : 4 corners of the code
    """
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                                                       parameters=arucoParams
                                                       )
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...(6,5,0)
        objp = np.zeros((2 * 2, 3), np.float32)
        objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)

        # Get image size
        img_size = (image.shape[1], image.shape[0])

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners[0][0]], img_size, None, None)

        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners[0][0], mtx, dist)
        print('degrees')
        print(np.degrees(rvecs[2]))
        # # project 3D points to image plane
        # imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        # image = draw(image, corners[0][0], imgpts)
        #
        image = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

        return ids[0], image, corners[0]
    else:
        return -1, image, []


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img

def get_aruco_coordinate_system(aligned_depth_frame, corners, degrees = True):
    """

    :param aligned_depth_frame:
    :param corners:
    :param degrees:
    :return:
    """
    coordinates = []
    for corner in corners[0]:
        x, y, z = convert_depth_pixel_to_metric_coordinate(aligned_depth_frame, corner[0], corner[1])
        point = np.array((x, y, z))
        coordinates.append(point)

    plane = Plane.best_fit(np.array(coordinates))
    rot_x = plane.normal[1]
    rot_y = plane.normal[0]
    rot_z = plane.normal[2]
    # rot_z = math.atan((coordinates[1][1] - coordinates[0][1])/(coordinates[1][0] - coordinates[0][0]))

    if degrees:
        return np.array([coordinates[0][0], coordinates[0][1], coordinates[0][2],
                         np.degrees(rot_x), np.degrees(rot_y), np.degrees(rot_z)])
    else:
        return np.array([coordinates[0][0], coordinates[0][1], coordinates[0][2], rot_x, rot_y, rot_z])


def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y):
    """
    	Convert the depth and image point information to metric coordinates
    	Parameters:
    	-----------
    	depth 	 	 	 : double
    						   The depth value of the image point
    	pixel_x 	  	 	 : double
    						   The x value of the image coordinate
    	pixel_y 	  	 	 : double
    							The y value of the image coordinate
    	Return:
    	----------
    	X : double
    		The x value in meters
    	Y : double
    		The y value in meters
    	Z : double
    		The z value in meters
    	"""

    camera_intrinsics = depth.profile.as_video_stream_profile().intrinsics

    Z = depth.get_distance(pixel_x, pixel_y)

    # Devuelve la coordenada respecto al centro de la camara
    # X = (Z * (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx) - 0.0175
    X = (Z * (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx)
    Y = Z * (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy
    return X, Y, Z
