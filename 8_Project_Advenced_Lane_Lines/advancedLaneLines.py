import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


def color_gradient(img, color_thresh=(150, 255), sobel_thresh=(20, 100)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= color_thresh[0]) & (s_channel <= color_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 10

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # TO-DO: Set the area of search based on activated x-values ###
    # within the +/- margin of our polynomial function ###
    # Hint: consider the window areas for the similarly named variables ###
    # in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 0, 255))

    # Draw green area of the lane
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(window_img, np.int_([pts]), (0, 255, 0))

    result = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)

    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # End visualization steps ##

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 830  # meters per pixel in x dimension
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    curvature = cal_curvature(ploty, left_fit_cr, right_fit_cr)

    left_point = left_fitx[-1]
    right_point = right_fitx[-1]

    offset = (right_point - left_point) * xm_per_pix - 3.7


    temp = np.copy(result)
    text1 = "Radius of Curvature = " + str(curvature)
    cv2.putText(temp, text1, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    text2 = "Vehicle Offset = " + str(abs(offset))
    cv2.putText(temp, text2, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    save_location = 'pipeline_imgs/curvature.jpg'
    mpimg.imsave(save_location, temp)

    return result, curvature, offset


def cal_curvature(ploty, left_fit, right_fit):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 830  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    curvature = (left_curverad + right_curverad) / 2
    return curvature


def pipeLine(img):
    # Read camera calibration information
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undistorted = undistort(img, mtx, dist)

    b, g, r = cv2.split(undistorted)
    undistorted = cv2.merge([r, g, b])

    save_location = 'pipeline_imgs/[1]undistorted.jpg'
    mpimg.imsave(save_location, undistorted)

    # Apply each of the thresholding functions
    udt = np.copy(undistorted)
    color_binary, combined_binary = color_gradient(udt, (150, 255), (20, 100))

    save_location = 'pipeline_imgs/[2]combined_binary.jpg'
    mpimg.imsave(save_location, combined_binary, cmap="gray")

    # perspective transformation
    src_points = np.float32([[img.shape[1], img.shape[0] - 10],
                             [0, img.shape[0] - 10],
                             [546, 460],
                             [732, 460]])
    des_points = np.float32([[img.shape[1], img.shape[0]],
                             [0, img.shape[0]],
                             [0, 0],
                             [img.shape[1], 0]])

    print(src_points)
    print(des_points)

    show_src_points = np.copy(undistorted)
    cv2.line(show_src_points, (src_points[0][0], src_points[0][1]), (src_points[1][0], src_points[1][1]), [255, 0, 0], 5)
    cv2.line(show_src_points, (src_points[1][0], src_points[1][1]), (src_points[2][0], src_points[2][1]), [255, 0, 0], 5)
    cv2.line(show_src_points, (src_points[2][0], src_points[2][1]), (src_points[3][0], src_points[3][1]), [255, 0, 0], 5)
    cv2.line(show_src_points, (src_points[3][0], src_points[3][1]), (src_points[0][0], src_points[0][1]), [255, 0, 0], 5)

    save_location = 'pipeline_imgs/[3]show_src_pts.jpg'
    mpimg.imsave(save_location, show_src_points)

    warped = warper(combined_binary, src_points, des_points)

    save_location = 'pipeline_imgs/[4]warped.jpg'
    mpimg.imsave(save_location, warped, cmap="gray")

    show_des_points = np.copy(warped)
    cv2.line(show_des_points, (des_points[0][0], des_points[0][1]), (des_points[1][0], des_points[1][1]), [255, 0, 0], 5)
    cv2.line(show_des_points, (des_points[1][0], des_points[1][1]), (des_points[2][0], des_points[2][1]), [255, 0, 0], 5)
    cv2.line(show_des_points, (des_points[2][0], des_points[2][1]), (des_points[3][0], des_points[3][1]), [255, 0, 0], 5)
    cv2.line(show_des_points, (des_points[3][0], des_points[3][1]), (des_points[0][0], des_points[0][1]), [255, 0, 0], 5)

    save_location = 'pipeline_imgs/[5]show_des_pts.jpg'
    mpimg.imsave(save_location, show_des_points, cmap="gray")

    out_img, left_fit, right_fit = fit_polynomial(warped)

    save_location = 'pipeline_imgs/[6]find_lane_pixels.jpg'
    mpimg.imsave(save_location, out_img, cmap="gray")

    result, curvature, offset = search_around_poly(warped, left_fit, right_fit)

    save_location = 'pipeline_imgs/[7]find_lane.jpg'
    mpimg.imsave(save_location, result)

    vers_warp = warper(result, des_points, src_points)

    save_location = 'pipeline_imgs/[8]vers_warp.jpg'
    mpimg.imsave(save_location, vers_warp)

    blend_onto_road = cv2.addWeighted(img, 1., vers_warp, 0.8, 0)

    b, g, r = cv2.split(blend_onto_road)
    blend_onto_road = cv2.merge([r, g, b])
    save_location = 'pipeline_imgs/[9]blend_onto_road.jpg'
    mpimg.imsave(save_location, blend_onto_road)

    text1 = "Radius of Curvature = " + str(curvature)
    cv2.putText(blend_onto_road, text1, (150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    text2 = "Vehicle Offset = " + str(abs(offset))
    cv2.putText(blend_onto_road, text2, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return blend_onto_road


def test_images():
    # img = cv2.imread('test_images/test1.jpg')

    images = os.listdir("test_images/")

    for this_image in images:
        location = 'test_images/' + this_image

        # location = 'test_images/curve.jpg'

        # reading in an image
        image = mpimg.imread(location)

        output_img = pipeLine(image)

        save_location = 'output_images/' + this_image
        mpimg.imsave(save_location, output_img)

        print("Finish Processing ", this_image)


def test_videos():
    videos = os.listdir("test_videos/")

    for this_video in videos:
        # location = 'test_videos/challenge.mp4'
        # white_output = 'test_videos_output/challenge.mp4'
        location = 'test_videos/' + this_video
        white_output = 'output_videos/' + this_video

        # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        # To do so add .subclip(start_second,end_second) to the end of the line below
        # Where start_second and end_second are integer values representing the start and end of the subclip
        # You may also uncomment the following line for a subclip of the first 5 seconds
        # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

        clip1 = VideoFileClip(location)
        white_clip = clip1.fl_image(pipeLine)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)

        print("Finish Processing ", this_video)


def test_single_image():
    img = cv2.imread('test_images/test1.jpg')

    output_img = pipeLine(img)

    # Plot to see results
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(output_img)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


def test_camera_cal():
    img = cv2.imread('camera_cal/calibration1.jpg')
    dist_pickle = pickle.load(open("camera_cal/wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    undistorted = undistort(img, mtx, dist)

    plt.imshow(undistorted)
    plt.show()

    save_location = 'output_images/calibration_check'
    mpimg.imsave(save_location, undistorted)


if __name__ == "__main__":
    test_single_image()
