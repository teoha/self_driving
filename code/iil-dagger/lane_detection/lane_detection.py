import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # <-- This line altered for grayscale.
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def contain_lanes(image):
    height,width,channels=image.shape
    region_of_interest_vertices = [
        (0, height),
        (0, height/2),
        (width/2, height/6),
        (width, height/2),
        (width, height),
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # Moved the cropping operation to the end of the pipeline.
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )


    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=350,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    steep_lines=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.4: # <-- Only consider extreme slope
                continue
            steep_lines.append(line)

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
                        
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.4: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = int(image.shape[0] * (1 / 7)) # <-- Just below the horizon
    max_y = int(image.shape[0]) # <-- The bottom of the image
    fitted_lines=[]
    if len(left_line_x)>0:
        isVisible_left=True
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))
        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
        fitted_lines.append([left_x_start, max_y, left_x_end, min_y])
    if len(right_line_x)>0:
        isVisible_right=True
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))
        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
        fitted_lines.append([right_x_start, max_y, right_x_end, min_y])

    if len(fitted_lines)==2:
        line_image = draw_lines(image, steep_lines) # <---- Add this call.
        plt.figure()
        plt.imshow(line_image)
        plt.show()
        line_image = draw_lines(
            image,
            [fitted_lines],
            thickness=5
        )
        plt.figure()
        plt.imshow(line_image)
        plt.show()
        input()

        return True

    return False