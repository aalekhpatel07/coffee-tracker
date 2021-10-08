import sys

import numpy as np
import cv2 as cv


def detect_labels(img: np.ndarray):
    """Detect the y-axis of the yellow labels on the coffee pot.
    """
    
    # Create a range of allowed colors.
    lower_color = np.array([20, 50, 0])
    upper_color = np.array([255, 255, 255])

    # Keep the pixels that lie within the range.
    color_filtered = cv.inRange(
        cv.cvtColor(img, cv.COLOR_RGB2HSV),
        lower_color,
        upper_color
    )
    
    # Keeping only the really bright pixels (converted to 255), change the dull ones to 0.
    # Helps distinguish the labels from other dull colors.
    _, thresholded = cv.threshold(color_filtered, 254, 255, cv.THRESH_BINARY)

    # Reduce the thickness of regions. Every 30x30 sliding window of 255 in the image gets replaced by a white pixel.
    # The stronger the erosion, the more the noise is removed, with a chance of removal of good pixels as well.
    eroded = cv.erode(thresholded, np.ones((30, 30)))

    # Now find outlines of the bright regions that remain after the thickness reduction.
    contours, _ = cv.findContours(eroded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Identify the contours that represent our labels.
    # Gotta be the two largest ones in terms of area.
    contour_areas = [(cv.contourArea(c), idx) for (idx, c) in enumerate(contours)]

    contour_largest_idx = max(contour_areas)[1]
    contour_second_largest_idx = max(filter(lambda item: item[1] != contour_largest_idx, contour_areas))[1]

    # Since the labels are sorta rectangular, find the mean of the contours' y-axes to approximate the vertical center of the labels.
    largest_vertical_center = np.mean(contours[contour_largest_idx][:, :, 1])
    second_largest_vertical_center = np.mean(contours[contour_second_largest_idx][:, :, 1])

    # Higher center implies the value is more towards the bottom of the image, and hence the vertical center of the bottom label.
    bottom_label = min(largest_vertical_center, second_largest_vertical_center)
    
    # Lower center implies the value is more towards the top of the image, and hence the vertical center of the top label.
    top_label = max(largest_vertical_center, second_largest_vertical_center)

    return bottom_label, top_label


def detect_coffee_level(img: np.ndarray, bottom_label=None, top_label=None):
    """Given an image of the coffee pot from our kitchen,
    estimate the percentage of coffee remaining in it.
    """

    # First detect all edges. Hopefully the coffee level line shows up as an edge.
    edges = cv.Canny(img, 100, 200)

    # Either we pre-compute and pass it, or we compute it right here.
    if bottom_label is None or top_label is None:
        # Get the position of the top and bottom labels.
        bottom_label, top_label = detect_labels(img)

    # Zoom into the interesting region since the coffee level line will always be in this region.
    region_of_interest = edges[round(bottom_label):round(top_label + 1), :]

    # Compute vertical and horizontal gradients using a Scharr filter.
    dy, dx = cv.Scharr(region_of_interest, cv.CV_8U, 1, 0), cv.Scharr(region_of_interest, cv.CV_8U, 0, 1)
    
    # Keep those pixels that have a strong horizontal and a weak vertical gradient.
    _, dx_thresholded = cv.threshold(dx, 254, 255, cv.THRESH_BINARY)
    _, dy_thresholded = cv.threshold(dy, 0, 1, cv.THRESH_BINARY)

    dx_thresholded = dx_thresholded.astype(dtype=np.uint8)
    dy_thresholded = dy_thresholded.astype(dtype=np.uint8)

    potential_horizontal_lines = np.bitwise_and(dx_thresholded, dy_thresholded)

    # List the non-dark pixels out of this collection of potential horizontal lines.
    point_set = cv.findNonZero(potential_horizontal_lines)

    # If the coffee level line is long enough, there will be a cluster of points in the
    # point_set that have a very close y-coordinates. Let's say we look at the 60th
    # percentile of the y-coordinates of the point_set. Then regardless of the coffee level
    # position, hopefully, the value at this percentile will correspond to some point on the actual line.

    # This should be pretty close to the actual
    # coffee level line assuming the line is long enough.
    point_set_y_coord_60th_percentile = np.percentile(point_set[:, :, 1], 60)

    # Only keep the points of that cluster (i.e. those that are within some threshold of the percentile value).
    points_around_the_actual_line = point_set[np.abs(point_set[:, :, 1] - point_set_y_coord_60th_percentile) < 5]

    # Estimate a horizontal line on those points.
    # It has the form: y = 0 * x + y_constant
    y_constant = np.polyfit(points_around_the_actual_line[:, 0].flatten(), points_around_the_actual_line[:, 1].flatten(), deg=0)[0]
    
    # Now we know the y-axis of the top label, the bottom label, as well as the estimated coffee level line.
    # This helps us compute the percentage of coffee remaining in the pot.
    percentage = y_constant / (top_label - bottom_label) * 100

    return percentage


def main(*img_paths):
    for img_path in img_paths:
        try:
            img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
            
            bottom_label, top_label = detect_labels(img)
            percentage = detect_coffee_level(img, bottom_label=bottom_label, top_label=top_label)

            print(img_path, f"{percentage}%")
        except Exception as e:
            print("Maybe file path invalid? idk. too lazy to handle exceptions rn.")
            print("Also, this algorithm is highly sensitive to the exact environment our coffee pot belongs to, so any results for other images will be pure garbage.")
            print("Actual Error:", e)
            print("Sample usage: python ./covfefe.py ./images/IMG_0323.jpg ")

if __name__ == '__main__':
    # img_path = './images/IMG_0323.jpg'
    if len(sys.argv) <= 1:
        print("Sample usage: python ./covfefe.py ./images/IMG_0323.jpg ")
    else:
        main(*sys.argv[1:])

