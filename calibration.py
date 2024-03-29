import pyrealsense2 as rs
import numpy as np
import cv2
from skspatial.objects import Plane
import math
from modules import tcp_server
from scipy.spatial.transform import Rotation as R
from modules.realsense import realsense


def main():

    camposes = []
    robot_poses = []

    # Chesboard
    nx = 11
    ny = 8
    first_image = True

    # init realsense
    rs = realsense()

    # Constants
    EnableTcpServer = False
    # Hay que cambiar la coordenada, si es hand eye la ref es respecto a la hoja de calib, sino es respecto a la coord de la camara
    hand_eye = True
    IP = "192.168.101.145"
    # IP = "localhost"
    Port = 6000

    if EnableTcpServer:
        # Arrancamos el servidor tcp coms robot
        tcp = tcp_server.TcpServer(IP, Port)
        tcp.start()

    # Streaming loop
    while True:
        
        # get frame
        aligned_depth_frame, aligned_color_image = rs.wait_for_frame()

        if not first_image:
            
            # get calibration
            ret, img, points, rvects = rs.checkerboard_calibration(aligned_color_image, nx, ny)
            
            # array to store coordinates
            coord = []

            if ret:
                if EnableTcpServer:

                    # Esperamos la pose del robot
                    robot_pose = tcp.waitForPose()
                    robot_pose = np.array(robot_pose.split(','), float)
                    robot_pose[0:3] = robot_pose[0:3] * 1000
                    robot_poses.append([robot_pose[0], robot_pose[1], robot_pose[2], robot_pose[3],robot_pose[4],robot_pose[5]])

                for i in range(nx * ny):
                    x, y, z = rs.convert_depth_pixel_to_metric_coordinate(aligned_depth_frame, points[0][i][0][0], points[0][i][0][1])
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


            cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Align Example', img)
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


if __name__ == '__main__':
    main()
