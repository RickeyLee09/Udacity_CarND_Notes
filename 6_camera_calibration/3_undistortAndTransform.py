import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_images/test_image2.png')
nx = 8  # the number of inside corners in x
ny = 6  # the number of inside corners in y


# plt.imshow(img)
# plt.show()


# MODIFY THIS FUNCTION TO GENERATE OUTPUT
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    # 2) Convert to grayscale
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)

    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # print(corners)

    # 4) If corners found:
    if ret == True:
        # a) draw corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
        src_points = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

        offset = 100
        img_size = (img.shape[1], img.shape[0])

        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        # dst_points = np.float32([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
        dst_points = np.float32([[offset, offset], [img_size[0] - offset, offset],
                                 [img_size[0] - offset, img_size[1] - offset],
                                 [offset, img_size[1] - offset]])

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        print(img_size)
        warped = cv2.warpPerspective(undistort, M, img_size, flags=cv2.INTER_LINEAR)

    # #delete the next two lines
    # M = None
    # warped = np.copy(img)
    return warped, M


top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
