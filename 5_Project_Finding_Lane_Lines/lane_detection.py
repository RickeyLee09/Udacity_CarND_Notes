# importing some useful packages
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math

from moviepy.editor import VideoFileClip
from IPython.display import HTML


# %matplotlib inline

# if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    right_lanes = []
    left_lanes = []

    for line in lines:
        # print(line)

        x1 = line[0][0]
        y1 = line[0][1]
        x2 = line[0][2]
        y2 = line[0][3]

        slope = (y2 - y1) / (x2 - x1)

        # print(slope)

        if 0.68 < slope:
            # Right lane
            right_lanes.append(line)
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            # print(line)
        elif -0.68 > slope:
            # Left lane
            left_lanes.append(line)
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        else:
            continue

    temp_right = np.array(right_lanes, dtype=np.int32)

    if temp_right.shape[0] == 0:
        return

    temp_right_reshape = temp_right.reshape(int(temp_right.shape[0]*2), int(temp_right.shape[2]/2))

    [vx, vy, x, y] = cv2.fitLine(temp_right_reshape, cv2.DIST_L2, 0, 0.01, 0.01)

    k = vy / vx
    b = y - k * x

    y1 = int((img.shape[1] / 18) * 7)
    x1 = (y1 - b) / k
    y2 = img.shape[1]
    x2 = (y2 - b) / k

    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    # Left Lane
    temp_left = np.array(left_lanes, dtype=np.int32)

    if temp_left.shape[0] == 0:
        return

    temp_left_reshape = temp_left.reshape(int(temp_left.shape[0]*2), int(temp_left.shape[2]/2))

    [vxl, vyl, xl, yl] = cv2.fitLine(temp_left_reshape, cv2.DIST_L2, 0, 0.01, 0.01)

    kl = vyl / vxl
    bl = yl - kl * xl

    y1l = int((img.shape[1] / 18) * 7)
    x1l = (y1l - bl) / kl
    y2l = img.shape[1]
    x2l = (y2l - bl) / kl

    cv2.line(img, (x1l, y1l), (x2l, y2l), color, thickness)

    # for x1, y1, x2, y2 in line:
    #     slope = (y2 - y1) / (x2 - x1)
    #     print(slope)
    #     cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    # plt.imshow(line_img)
    # plt.show()

    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def color_selection(img, red_threshold, green_threshold, blue_threshold):
    color_select = np.copy(img)
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]
    color_threshold = (img[:, :, 0] < rgb_threshold[0]) | \
                      (img[:, :, 1] < rgb_threshold[1]) | \
                      (img[:, :, 2] < rgb_threshold[2])
    color_select[color_threshold] = [0, 0, 0]
    return color_select, color_threshold


def pipeLine(image):
    ysize = image.shape[0]
    xsize = image.shape[1]

    color_img = np.copy(image)
    line_img = np.copy(image)
    color_output_img = np.copy(image)

    vertices = [np.array([[(xsize/14)*2, ysize], [(xsize/14)*13, ysize], [xsize/2, ysize/2]], np.int32)]
    line_img = region_of_interest(image, vertices)

    color_img, color_threshold = color_selection(line_img, 180, 180, 100)

    color_output_img[~color_threshold] = [255, 0, 0]

    gray_img = grayscale(color_output_img)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_grey = gaussian_blur(gray_img, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 120
    high_threshold = 140

    edges = canny(blur_grey, low_threshold, high_threshold)

    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 10  # angular resolution in radians of the Hough grid
    threshold = 30  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 3  # minimum number of pixels making up a line
    max_line_gap = 15  # maximum gap in pixels between connectable line segments
    # line_image = np.copy(image) * 0  # creating a blank to draw lines on

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

    result = weighted_img(line_image, image, 0.8, 1, 0)

    # plt.imshow(masked_edges)
    # plt.show()

    return result


def test_images():
    images = os.listdir("test_images/")

    for this_image in images:
        location = 'test_images/' + this_image

        # location = 'test_images/curve.jpg'

        # reading in an image
        image = mpimg.imread(location)

        output_img = pipeLine(image)

        # plt.imshow(output_img)
        # plt.show()

        save_location = 'test_images_output/' + this_image
        mpimg.imsave(save_location, output_img)


def test_videos():
    videos = os.listdir("test_videos/")

    for this_video in videos:
        # location = 'test_videos/challenge.mp4'
        # white_output = 'test_videos_output/challenge.mp4'
        location = 'test_videos/' + this_video
        white_output = 'test_videos_output/' + this_video

        # To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
        # To do so add .subclip(start_second,end_second) to the end of the line below
        # Where start_second and end_second are integer values representing the start and end of the subclip
        # You may also uncomment the following line for a subclip of the first 5 seconds
        # clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

        clip1 = VideoFileClip(location)
        white_clip = clip1.fl_image(pipeLine)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    test_images()
    test_videos()
