import socket
import sys


class TcpServer:
    def __init__(self, IP, Port):

        #Instancies
        self.IP = IP
        self.Port = Port

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #Force close
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Bind the socket to the port
        server_address = (self.IP, self.Port)
        print('starting up on {} port {}'.format(*server_address))
        self.sock.bind(server_address)

    def __del__(self):
        self.sock.close()

    def start(self):
    
        # Listen for incoming connections
        self.sock.listen(1)
        
        # Wait for a connection
        print('waiting for a connection')
        self.connection, self.client_address = self.sock.accept()
        print('connection from', self.client_address)

    def waitForPose(self):
        data = self.connection.recv(256)

        # Quitamos los caracteres binarios
        tag = str(data) [4:len(data)+1]

        return tag

    def send(self, msg):
        self.connection.sendall(bytes(msg, encoding= 'utf-8'))

    def close(self):
        self.sock.close()
