import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg

# # Read in the saved objpoints and imgpoints
# dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
# objpoints = dist_pickle["objpoints"]
# imgpoints = dist_pickle["imgpoints"]
#
# # Read in an image
# img = cv2.imread('test_images/test_image.png')


def getPoints():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('calibration_wide/GO*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8, 6), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            print(fname)

    print("read images done!")
    cv2.destroyAllWindows()
    return objpoints, imgpoints


def calibration(img, objpoints, imgpoints):
    # Test undistortion on an image
    # img = cv2.imread('calibration_wide/test_image.jpg')
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {"mtx": mtx, "dist": dist}
    pickle.dump(dist_pickle, open("calibration_wide/wide_dist_pickle.p", "wb"))

    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return dst


# TODO: Write a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and
# returns the undistorted image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    undist = np.copy(img)  # Delete this line
    return undist


def main():
    img = cv2.imread('test_images/test_image.png')
    objpoints, imgpoints = getPoints()

    undistorted = calibration(img, objpoints, imgpoints)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()


if __name__ == "__main__":
    main()
