import math
import cv2
import numpy as np
import sys


class RoboGambit_Perception:

    def __init__(self):
        # PARAMETERS - Camera intrinsics provided by organisers (DO NOT MODIFY)
        self.camera_matrix = np.array([
            [1030.4890823364258, 0, 960],
            [0, 1030.489103794098, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((1, 5))

        # INTERNAL VARIABLES
        self.corner_world = {
            21: (350, 350),
            22: (350, -350),
            23: (-350, -350),
            24: (-350, 350)
        }
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        self.H_matrix = None

        self.board = np.zeros((6, 6), dtype=int)

        # ARUCO DETECTOR
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict,self.aruco_params)

        print("Perception Initialized")


    # DO NOT MODIFY THIS FUNCTION
    def prepare_image(self, image):
        """
        DO NOT MODIFY.
        Performs camera undistortion and grayscale conversion.
        """
        undistorted_image = cv2.undistort(image,self.camera_matrix,self.dist_coeffs,None,self.camera_matrix)
        gray_image = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
        return undistorted_image, gray_image


    # TODO: IMPLEMENT PIXEL → WORLD TRANSFORMATION
    def pixel_to_world(self, pixel_x, pixel_y):

        if self.H_matrix is None:
            return None, None

        pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)

        world_pt = cv2.perspectiveTransform(pt, self.H_matrix)

        x_world = world_pt[0][0][0]
        y_world = world_pt[0][0][1]

        return x_world, y_world


    # PARTICIPANTS MODIFY THIS FUNCTION
    def process_image(self, image):
        """
        Main perception pipeline.
        Participants must implement:
        - ArUco detection
        - Homography computation
        - Pixel → world conversion
        - Board reconstruction
        """

        self.board[:] = 0

        # Preprocess image (Do not modify)
        undistorted_image, gray_image = self.prepare_image(image)

        # corners, ids, rejected = self.detector.detectMarkers(gray_image)
        # if ids is not None:
        #     cv2.aruco.drawDetectedMarkers(undistorted_image,corners,ids)
        corners, ids, rejected = self.detector.detectMarkers(gray_image)

        if ids is None:
            return

        cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)


        # TODO: Extract corner marker pixels

        # Identify markers with IDs 21–24
        # Store their pixel centers
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.corner_world:
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:,0])
                center_y = np.mean(marker_corners[:,1])
                self.corner_pixels[marker_id] = (center_x, center_y)

        # TODO: Build pixel and world matrices

        # Use detected corner markers and
        # known world coordinates
        if len(self.corner_pixels) == 4:

            self.pixel_matrix = []
            self.world_matrix = []

            for marker_id in self.corner_world:
                px, py = self.corner_pixels[marker_id]
                wx, wy = self.corner_world[marker_id]
                self.pixel_matrix.append([px, py])
                self.world_matrix.append([wx, wy])
            self.pixel_matrix = np.array(self.pixel_matrix, dtype=np.float32)
            self.world_matrix = np.array(self.world_matrix, dtype=np.float32)

        # TODO: Compute homography matrix

        # Use:
        # cv2.findHomography()
        self.H_matrix, _ = cv2.findHomography(self.pixel_matrix, self.world_matrix)
        


            

        # TODO: Convert piece markers to world coordinates

        # For each marker with ID 1–10:
        # 1. Compute center pixel
        # 2. Convert to world using pixel_to_world()
        # 3. Call place_piece_on_board()

        
        for i, marker_id in enumerate(ids.flatten()):
            if 1 <= marker_id <= 10:
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:,0])
                center_y = np.mean(marker_corners[:,1])
                x_world, y_world = self.pixel_to_world(center_x, center_y)
             
                
                if x_world is not None:
                    self.place_piece_on_board(marker_id, x_world, y_world)

        # Visualization (Do not modify)
        res = cv2.resize(undistorted_image, (1152,648))
        cv2.imshow("Detected Markers", res)
        self.visualize_board()


    # TODO: IMPLEMENT BOARD PLACEMENT
    def place_piece_on_board(self, piece_id, x_coord, y_coord):

        """
        Places detected piece on the closest board square.

        Board definition:

        6x6 grid
        top-left corner = (300,300)
        square size = 100mm
        """

        # Write your code here
       

        top_left_x = 300
        top_left_y = 300
        cell_size = 100

        col = int((top_left_x - x_coord) / cell_size)
        row = int((top_left_y - y_coord) / cell_size)

        if 0 <= row < 6 and 0 <= col < 6:
            self.board[row][col] = piece_id

  


    # DO NOT MODIFY THIS FUNCTION
    def visualize_board(self):
        """
        Draw a simple 6x6 board with detected piece IDs
        """
        cell_size = 80
        board_img = np.ones((6*cell_size,6*cell_size,3),dtype=np.uint8) * 255

        for r in range(6):
            for c in range(6):
                x1 = c*cell_size
                y1 = r*cell_size
                x2 = x1+cell_size
                y2 = y1+cell_size
                cv2.rectangle(board_img,(x1,y1),(x2,y2),(0,0,0),2)

                piece = int(self.board[r][c])
                if piece != 0:
                    cv2.putText(board_img,str(piece),(x1+25,y1+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        cv2.imshow("Game Board", board_img)


# DO NOT MODIFY
def main():
    # To run code, use python/python3 perception.py path/to/image.png
    if len(sys.argv) < 2:
        print("Usage: python perception.py image.png")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    perception = RoboGambit_Perception()
    perception.process_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
