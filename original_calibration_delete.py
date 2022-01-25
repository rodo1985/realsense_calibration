import pyrealsense2 as rs
import numpy as np
import cv2
from skspatial.objects import Plane
import math
from modules import tcp_server
from scipy.spatial.transform import Rotation as R


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


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
        img = draw(img, corners, imgpts)
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

    return ret, img, imgpoints, rvecs


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
    # X = (Z * (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx)
    X = (Z * (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx) - 0.0175
    Y = Z * (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy
    return X * 1000, Y * 1000, Z * 1000


def main():

    camposes = []
    robot_poses = []

    # Chesboard
    nx = 11
    ny = 8
    first_image = True


    # Constants
    EnableTcpServer = False
    # Hay que cambiar la coordenada, si es hand eye la ref es respecto a la hoja de calib, sino es respecto a la coord de la camara
    hand_eye = True
    IP = "192.168.101.145"
    # IP = "localhost"
    Port = 6000

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth stream
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    align_to = rs.stream.color
    align = rs.align(align_to)

    if EnableTcpServer:
        # Arrancamos el servidor tcp coms robot
        tcp = tcp_server.TcpServer(IP, Port)
        tcp.start()

    # Streaming loop
    try:
        while True:
            
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            if not first_image:

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                bg_removed = color_image

                ret, img, points, rvects = checkerboard_calibration(bg_removed, nx, ny)

                coord = []

                if ret:
                    if EnableTcpServer:
                        # Esperamos la pose del robot
                        robot_pose = tcp.waitForPose()
                        robot_pose = np.array(robot_pose.split(',q'), float)
                        robot_pose[0:3] = robot_pose[0:3] * 1000
                        robot_poses.append([robot_pose[0], robot_pose[1], robot_pose[2], robot_pose[3],robot_pose[4],robot_pose[5]])

                    for i in range(nx * ny):

                        x, y, z = convert_depth_pixel_to_metric_coordinate(aligned_depth_frame, points[0][i][0][0],
                                                                        points[0][i][0][1])
                        point = np.array((x, y, z))
                        if point[0] != 0 and point[1] != 0 and point[2] != 0:
                            coord.append(point)

                    if(len(coord) > 2):
                        plane = Plane.best_fit(np.array(coord))
                        rot_x = math.acos(plane.normal[0]) + math.pi/2
                        rot_y = math.acos(plane.normal[1]) - math.pi/2
                        rot_z = math.pi / 2 + rvects[2][0]
                        
                        euler = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=False)
                        quaternion = euler.as_quat()

                        if hand_eye:
                            
                            transform = np.eye(3)
                            transform[:3, :3] = euler.inv().as_matrix()

                            p = np.array([coord[0][0], coord[0][1], coord[0][2]]) * -1

                            transformed_point = np.dot(p, transform)

                            camposes.append([transformed_point[0], transformed_point[1], transformed_point[2], quaternion[3], quaternion[0], quaternion[1], quaternion[2]
                                            ])
                        else:
                            camposes.append([coord[0][0], coord[0][1], coord[0][2], quaternion[3], quaternion[0], quaternion[1], quaternion[2]
                                        ])

                        if not EnableTcpServer:
                            print(camposes[len(camposes)-1])
                    else:
                        print('Not enough coordinates')

                    bg_removed = img

                cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Align Example', bg_removed)
                key = cv2.waitKey()

                # # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    if EnableTcpServer:
                        # # Guardamos las coordenadas
                        # np.savetxt('test.txt', camposes, delimiter=',')  # X is an array
                        for i in range(len(camposes)):
                            print(str(robot_poses[i][0]) + ' ' +
                                str(robot_poses[i][1]) + ' ' +
                                str(robot_poses[i][2]) + ' ' +
                                str(robot_poses[i][3]) + ' ' +
                                str(robot_poses[i][4]) + ' ' +
                                str(robot_poses[i][5]) + ' ' +
                                str(camposes[i][0]) + ' ' +
                                str(camposes[i][1]) + ' ' +
                                str(camposes[i][2]) + ' ' +
                                str(camposes[i][3]) + ' ' +
                                str(camposes[i][4]) + ' ' +
                                str(camposes[i][5]) + ' ' +
                                str(camposes[i][6])
                                )


                    if EnableTcpServer:
                        tcp.close()
                        del tcp

                    cv2.destroyAllWindows()
                    break
            else:
                first_image = False
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
