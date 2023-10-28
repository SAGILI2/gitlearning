import os
import cv2
import numpy as np
import math
import imutils
from scipy.ndimage import interpolation as inter


def get_line_length(pt1, pt2):
    """
    calculate line length
    :param pt1: start point of line
    :param pt2: end point of line
    :return: euclidean distance bw the 2 points
    """
    [x1, y1] = pt1
    [x2, y2] = pt2
    lin_len = np.power((np.power(x1 - x2, 2) + np.power(y1 - y2, 2)), 0.5)
    return lin_len


def getHorLines(im):
    """
    extract horizontal lines
    :param im:
    :return: end co-ordinates of identified lines
    """
    if len(im.shape) == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im
    gray = cv2.bitwise_not(gray)
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
    )
    horizontal = np.copy(bw)
    width = horizontal.shape[1]
    height = horizontal.shape[0]
    horizontalSize = width // 50  # min horizontal line length
    # Create structure element for extracting horizontal lines through morphology operations
    horStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horStructure)
    horizontal = cv2.dilate(horizontal, horStructure)
    contours = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    hor_lines = []
    for c in contours:
        sort_c = sorted(c, key=lambda x: x[0][1])
        pt1 = tuple(sort_c[0][0])
        pt2 = tuple(sort_c[-1][0])
        # check to verify extracted line is not a border
        if (
            height / 10 < pt1[1] < height - height / 10
            and height / 10 < pt2[1] < height - height / 10
        ):
            hor_lines.append([pt1, pt2])
        # cv2.line(im, pt1, pt2, (255, 0, 0), 2)
    return hor_lines


def get_alignment(img):
    """
    calculate the angle of rotation
    :param path: image path
    :return: orientation angle in degrees
    """
    print("getting align for shape", img.shape)
    hor_lines = getHorLines(img)
    img_width = img.shape[1]
    hor_lines = sorted(
        hor_lines, key=lambda x: get_line_length(x[0], x[1]), reverse=True
    )
    if (
        len(hor_lines) > 0
        and get_line_length(hor_lines[0][0], hor_lines[0][1]) > img_width / 10
    ):
        if len(hor_lines) > 5:
            hor_lines = hor_lines[:5]
        angles = []
        lengths = []
        im_out = img.copy()
        for h in hor_lines:
            cv2.line(im_out, h[0], h[1], (0, 0, 255), 4)
            [x1, y1] = h[0]
            [x2, y2] = h[1]
            # print(h)
            if x2 < x1:
                rotate = "clock"
            else:
                rotate = "anti"
            length = get_line_length(h[0], h[1])
            sin_val = abs(y2 - y1) / length
            angle = math.degrees(math.asin(sin_val))
            if "anti" in rotate:
                angle = -1 * angle
            # print("angle", angle,"rotate", rotate)
            angles.append(angle)
            lengths.append(length)
        if len(angles) > 0:
            rotAngle = np.average(angles, weights=lengths)
            if rotAngle != 0:
                return rotAngle

    return 0


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    scores = (hist[1:] - hist[:-1]) ** 3
    scores = [abs(s) for s in scores]
    score = np.sum(scores)
    score = abs(score)
    return hist, score


def get_rel_snippet(bin_img):
    print("BIN IMAGE SHAPE", bin_img.shape)
    [h, w] = bin_img.shape[:2]
    h_10 = int(0.10 * h)
    h_30 = int(0.30 * h)
    h_35 = int(0.35 * h)
    w_10 = int(0.15 * w)
    h_5 = int(0.05 * h)
    if sum(sum(bin_img[h_35 : (h - h_30), w_10 : (w - w_10)])) > 50000:
        return bin_img[h_35 : (h - h_30), w_10 : (w - w_10)]
    elif sum(sum(bin_img[h_10 : (h - h_35), w_10 : (w - w_10)])) > 50000:
        return bin_img[h_10 : (h - h_35), w_10 : (w - w_10)]
    elif sum(sum(bin_img[h_5 : (h - h_35), w_10 : (w - w_10)])) > 50000:
        return bin_img[h_10 : (h - h_35), w_10 : (w - w_10)]
    return bin_img[h_35 : (h - h_10), w_10 : (w - w_10)]


def orient(bin_img):
    if len(bin_img.shape) == 3:
        bin_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)

    # bin_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    [h, w] = bin_img.shape[:2]
    bin_img = get_rel_snippet(bin_img)
    bin_img = cv2.dilate(bin_img, np.ones((2, 2)))
    bin_img = bin_img / 255
    delta = 5
    limit = 45
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        # print("angle",angle,"score",score)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print("Best angle - first run: ", best_angle)

    delta = 1
    angles = np.arange(best_angle - 5, best_angle + 5, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(bin_img, angle)
        # print("angle",angle,"score",score)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    # deskew_im = imutils.rotate_bound(img,-1*best_angle)
    return -1 * best_angle
