import cv2
import os
import numpy as np
from utils.orientation_util import get_alignment, orient
import imutils
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import time
from doctr.get_text_boxes import detect_text
from collections import Counter


# from main.unet_model import get_mask


def smooth(x, window_len=11, window="hanning"):
    """
    Taken from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    # if x.ndim != 1:
    #     raise ValueError, "smooth only accepts 1 dimension arrays."
    #
    # if x.size < window_len:
    #     raise ValueError, "Input vector needs to be bigger than window size."
    #
    # if window_len < 3:
    #     return x
    #
    # if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    #     raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    # print(len(s))
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))


def overlaps_x_coord(ref_line, curr_line):
    [ref_x1, ref_y1, ref_w, ref_h] = ref_line
    [curr_x1, curr_y1, curr_w, curr_h] = curr_line
    ref_x2 = ref_x1 + ref_w
    ref_y2 = ref_y1 + ref_h
    curr_x2 = curr_x1 + curr_w
    curr_y2 = curr_y1 + curr_h
    if (
        (curr_x1 >= ref_x1 and curr_x1 <= ref_x2)
        or (curr_x2 >= ref_x1 and curr_x2 <= ref_x2)
        or (curr_x1 <= ref_x1 and curr_x2 >= ref_x2)
    ):
        overlap_val_x = overlap(curr_x1, curr_x2, ref_x1, ref_x2)
        overlap_val_y = overlap(curr_y1, curr_y2, ref_y1, ref_y2)
        # print("overlap val", overlap_val)
        if overlap_val_x > 5 and overlap_val_y < 0.85 * (curr_y2 - curr_y1):
            return True
    return False


def overlaps_x_coord_any(curr_line, ref_lines, common_height):
    for line in ref_lines:
        if (
            line[2] > 0.25 * common_height and line[3] > 0.7 * common_height
        ):  # min width of the contour to be considered
            if overlaps_x_coord(line, curr_line):
                return True
    return False


# Function to generate horizontal projection profile
def getHorizontalProjectionProfile(image):
    # Convert white spots to ones
    image[image == 255] = 1

    horizontal_projection = np.sum(image, axis=1)

    return horizontal_projection


def transform_coords(modified_roi_lines, y_min, x_min):
    res = []
    for line in modified_roi_lines:
        # line = [l["pts"] for l in line]
        modified_line = []
        for roi in line:
            id = roi["id"]
            [x, y, w, h] = roi["pts"]
            x = x_min + x
            y = y_min + y
            modified_line.append({"id": id, "pts": [x, y, w, h]})
        res.append(modified_line)
    return res


def is_separated_by_vertical_line(connecting_lines, vertical_lines, multiple=False):
    intersections = []
    if multiple:
        line_strings = [LineString(l) for l in connecting_lines]
    else:
        line = LineString(connecting_lines)
    for ver_line in vertical_lines:
        if multiple:
            other = LineString(ver_line)
            if all([line.intersects(other) for line in line_strings]):
                intersections.append(line_strings[0].intersection(other))
        else:
            other = LineString(ver_line)
            if line.intersects(other):
                return line.intersection(other)
    if len(intersections) > 0:
        intersections = sorted(intersections, key=lambda x: x.x)
        return intersections
    return False


def contains_same_heights_2(line):
    line = sorted(line, key=lambda x: x[3])
    first_height = line[0][3]
    last_height = line[-1][3]
    same_heights = True
    max_diff = max(3, 0.2 * min(first_height, last_height))
    # max_diff = 0.1 * min(first_height, last_height)
    if abs(first_height - last_height) > max_diff:
        # and not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
        same_heights = False
    return same_heights


def contains_same_heights(line):
    same_heights = True
    prev_height = line[0][3]
    prev_ymin = line[0][1]
    for block in line:
        curr_height = block[3]
        curr_ymin = block[1]
        # max_diff = 0.1 * min(curr_height, prev_height)
        max_diff = max(3, 0.2 * min(curr_height, prev_height))
        if abs(curr_height - prev_height) > max_diff:
            # and not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
            same_heights = False
            break
        # if not (curr_height < 0.6*prev_height and (abs(prev_ymin-curr_ymin) < 0.75*prev_height)):
        prev_height = block[3]
        prev_ymin = block[1]
    return same_heights


def sort_box_coords(box_coords):
    box_coords = sorted(box_coords, key=lambda x: x[0])
    if box_coords[0][1] < box_coords[1][1]:
        tl = box_coords[0]
        bl = box_coords[1]
    else:
        tl = box_coords[1]
        bl = box_coords[0]

    if box_coords[2][1] < box_coords[3][1]:
        tr = box_coords[2]
        br = box_coords[3]
    else:
        tr = box_coords[3]
        br = box_coords[2]
    return {"top_left": tl, "top_right": tr, "btm_right": br, "btm_left": bl}


def getCombinedRect(minAreaRects):
    if len(minAreaRects) > 0:
        minAreaRects = [sort_box_coords(s) for s in minAreaRects]
        minAreaRects = sorted(minAreaRects, key=lambda x: x["top_left"][0])
        return [
            minAreaRects[0]["top_left"],
            minAreaRects[0]["btm_left"],
            minAreaRects[-1]["top_right"],
            minAreaRects[-1]["btm_right"],
        ]
    return []


def combine(merge_words):
    x_mins = []
    y_mins = []
    x_maxs = []
    y_maxs = []
    text = ""
    word_ids = []
    minAreaRects = []
    for word in merge_words:
        [x_min, y_min, w, h] = word["pts"]
        x_max = x_min + w
        y_max = y_min + h
        x_mins.append(x_min)
        y_mins.append(y_min)
        x_maxs.append(x_max)
        y_maxs.append(y_max)
        if "text" in word.keys():
            text = text + " " + word["text"]
        word_ids.append(word["id"])
        if "minAreaRect" in word:
            minAreaRects.append(word["minAreaRect"])

    id_ = "-".join([str(i) for i in word_ids])
    if len(minAreaRects) == 1:
        minAreaRect_combined = minAreaRects[0]
    elif len(minAreaRects) > 1:
        minAreaRect_combined = getCombinedRect(minAreaRects)
    if len(minAreaRects) > 0:
        return {
            "text": text.strip(),
            "pts": [
                min(x_mins),
                min(y_mins),
                max(x_maxs) - min(x_mins),
                max(y_maxs) - min(y_mins),
            ],
            "ids": word_ids,
            "id": id_,
            "minAreaRect": minAreaRect_combined,
        }
    else:
        return {
            "text": text.strip(),
            "pts": [
                min(x_mins),
                min(y_mins),
                max(x_maxs) - min(x_mins),
                max(y_maxs) - min(y_mins),
            ],
            "ids": word_ids,
            "id": id_,
        }


def remove_bg(image, h, w, blur_window=51):
    # [h, w] = image.shape[:2]
    if blur_window != 51:
        blur_window = int(min(h / 4, w / 4))
        if blur_window % 2 == 0:
            blur_window += 1
    blur_bg = cv2.medianBlur(image, blur_window)
    # wo_bg = cv2.subtract(np.bitwise_not(image), np.bitwise_not(blur_bg))
    wo_bg = np.subtract(
        np.bitwise_not(image).astype(int), np.bitwise_not(blur_bg).astype(int)
    )
    wo_bg[wo_bg < -50] = 255
    wo_bg[wo_bg < 0] = 0
    wo_bg = wo_bg.astype(np.uint8)
    morphKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    wo_bg = cv2.morphologyEx(wo_bg, cv2.MORPH_CLOSE, morphKernel)
    return wo_bg


# def snippetTextDetection(img_path,coords):
#     [x1,y1,x2,y2] = coords
#     img = cv2.imread(img_path)
#     snippet = img[y1:y2,x1:x2]
#     canvas = snippet.copy()
#     width = img.shape[1]
#     height = img.shape[0]
#     adap_thresh_window = min(int(width / 5), int(height / 5))
#     if adap_thresh_window % 2 == 0:
#         adap_thresh_window += 1
#     if len(snippet.shape)==3:
#         gray = cv2.cvtColor(snippet, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = snippet
#     gray_wo_bg = remove_bg(gray,height,width, 51)  # remove any background color
#     blur = cv2.medianBlur(gray_wo_bg, 3)  # to remove background noise
#     morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, morphKernel)
#     dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
#     grad_dil = cv2.dilate(grad, dilate_kernel)
#     adap_thresh = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
#                                         cv2.THRESH_BINARY, adap_thresh_window, -2)
#     otsu_thresh = cv2.threshold(grad, 80, 255, cv2.THRESH_OTSU)[1]
#     fin_thresh = cv2.bitwise_and(otsu_thresh, adap_thresh)
#
#
#     rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 1))
#     gradX = cv2.morphologyEx(fin_thresh, cv2.MORPH_CLOSE, rectKernel)
#
#     thresh = cv2.threshold(gradX, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     rois = []
#     for idx, c in enumerate(cnts):
#         [x, y, w, h] = cv2.boundingRect(c)
#         if len(c) > 5 and (h > 10 or (h > 5 and w * h > 50)) and w > 0:
#             # cv2.drawContours(canvas,[c],-1,(0,0,255),2)
#             cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             rois.append({"id": idx, "pts": [x, y, w, h]})
#     lines = sort_contours(rois)
#     return [snippet,lines]


class TextDetection:
    """
    class to identify blobs of text in an image
    self.lines --> list of lists, where each inner list contains texts in a single row( sorted from left to right
                 contains the information of text and the coordinates of every contour that's identified
                 coordinates format -->(x,y,w,h)
                 (x,y) --> top left
                 w --> width
                 h --> height
    """

    def __init__(self, img, path, text_mask=None, debug=False):
        t1 = time.time()
        self.path = path
        self.debug = debug
        self.basename = os.path.basename(self.path)
        self.ID = self.basename
        ## unet_thresh = self.resize_image(cv2.imread(os.path.join("seg_res_apb",self.basename),0))
        if text_mask is None:
            # unet_thresh = self.resize_image(get_mask([cv2.imread(path,0)])[0])
            unet_thresh = cv2.imread("samples/Aadhar_3_mask.jpg", 0)
        else:
            unet_thresh = self.resize_image(text_mask)

        # unet_thresh = self.resize_image(text_mask)
        unet_thresh = cv2.threshold(unet_thresh, 125, 255, cv2.THRESH_BINARY)[1]
        # cv2.imwrite('text_masks/'+self.basename,unet_thresh)
        # self.modified_path = os.path.join("mod_jpg", os.path.basename(path))
        self.modified_path = path
        im_color = img
        self.img = self.increase_contrast(self.resize_image(img))

        # self.img = self.resize_image(img)
        # self.im_shape = self.img.shape
        # print('Time taken before alignment',time.time() - t1)
        st_t = time.time()
        # self.alignment = get_alignment(self.img)
        self.alignment = orient(unet_thresh)
        self.oriented_orig_img = imutils.rotate_bound(im_color, self.alignment)
        self.im_shape = self.oriented_orig_img.shape
        print("write self.modified_path {}".format(self.modified_path))
        cv2.imwrite(self.modified_path, self.oriented_orig_img)
        print("Time after verify lines 0", time.time() - st_t)
        print("self.modified_path is {}".format((os.path.abspath(self.modified_path))))
        self.regions = detect_text(os.path.abspath(self.modified_path))
        self.common_height = self.get_common_height(self.regions)
        self.lines = self.sort_contours(self.regions)
        self.lines = self.remove_noise()

        self.lines = self.merge_close_texts()
        self.add_padding()

        print("Time after verify lines", time.time() - st_t)
        print("FINAL TIME", time.time() - t1)

    def increase_contrast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ctr = np.bincount(gray.ravel())
        ctr = ctr[::-1]
        max_val = 0
        for i, v in enumerate(ctr):
            if v > 5:
                max_val = 255 - i
                break
        if max_val > 0:
            contrast_alpha = 255 / max_val
            beta = 20
            new_image = cv2.addWeighted(
                gray, contrast_alpha, np.zeros(gray.shape, gray.dtype), 0, beta
            )
            new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
            return new_image
        else:
            return img

    def get_dist(self, coord_1, coord_2):
        [x1, y1, w1, h1] = coord_1
        [x2, y2, w2, h2] = coord_2
        mid_pt_1 = [x1 + int(w1 / 2), y1 + int(h1 / 2)]
        mid_pt_2 = [x2 + int(w2 / 2), y2 + int(h2 / 2)]
        dist = np.power(
            np.power((mid_pt_1[0] - mid_pt_2[0]), 2)
            + np.power((mid_pt_1[1] - mid_pt_2[1]), 2),
            0.5,
        )
        return dist

    def get_min_dist(self, coord1, coord2):
        """
        get the minimum horizontal distance
        :param coord1: [x1,w1]
        :param coord2: [x2,w2]
        :return: distance
        """
        [x1, w1] = coord1
        [x2, w2] = coord2
        x1_max = x1 + w1
        x2_max = x2 + w2
        if x1 < x2 and x1_max < x2:
            return x2 - x1_max
        elif x1 > x2_max and x1_max > x2_max:
            return x1 - x2_max
        else:
            return min(
                abs(x1 - x2), abs(x1_max - x2_max), abs(x1 - x2_max), abs(x1_max - x2)
            )

    def get_closest_contour(self, curr_coords, line):
        if len(line) == 1:
            return line[0]
        else:
            line_rel = list(
                filter(
                    lambda x: (
                        (
                            x["pts"][3] >= 0.85 * self.common_height
                            or x["pts"][3] * x["pts"][2]
                            >= 0.75 * self.common_height * self.common_height
                        )
                        and x["pts"][2] > 2 * self.common_height
                    ),
                    line,
                )
            )
            if len(line_rel) > 0:
                distances = [self.get_dist(curr_coords, b["pts"]) for b in line_rel]
                return line_rel[np.argmin(distances)]
            else:
                return line[0]

    def is_vertically_close(self, curr_coords, prev_line):
        [x, y, w, h] = curr_coords
        y_max = y + h
        for b in prev_line:
            if overlaps_x_coord(curr_coords, b["pts"]):
                [b_x, b_y, b_w, b_h] = b["pts"]
                b_y_max = b_y + b_h
                if (y - b_y_max) < 0.4 * self.common_height:
                    return True
        return False

    def sort_box_coords(self, box_coords):
        box_coords = sorted(box_coords, key=lambda x: x[0])
        if box_coords[0][1] < box_coords[1][1]:
            tl = box_coords[0]
            bl = box_coords[1]
        else:
            tl = box_coords[1]
            bl = box_coords[0]

        if box_coords[2][1] < box_coords[3][1]:
            tr = box_coords[2]
            br = box_coords[3]
        else:
            tr = box_coords[3]
            br = box_coords[2]
        return {"top_left": tl, "top_right": tr, "btm_right": br, "btm_left": bl}

    def get_angle(self, left_coords, right_coords):
        [lx, ly] = left_coords
        [rx, ry] = right_coords
        angle = np.degrees(np.arctan((ly - ry) / (rx - lx)))
        return angle

    def get_poi(self, line, x_val):
        [x0, y0] = line[0]
        [x1, y1] = line[1]

        # linear equation: y = m*x + c
        m = (y1 - y0) / (x1 - x0)
        c = y0 - m * x0
        y = m * x_val + c
        return y

    def falls_in_the_area_of_line(self, curr_contour, closest_contour):
        if (
            "minAreaRect" in curr_contour.keys()
            and "minAreaRect" in closest_contour.keys()
        ):
            curr_contour_box_coords = self.sort_box_coords(curr_contour["minAreaRect"])
            closest_contour_box_coords = self.sort_box_coords(
                closest_contour["minAreaRect"]
            )
            if (
                closest_contour_box_coords["top_left"][0]
                > curr_contour_box_coords["top_right"][0]
                and closest_contour_box_coords["top_left"][0]
                > curr_contour_box_coords["top_left"][0]
            ):
                curr2closest_poi_btm = self.get_poi(
                    [
                        curr_contour_box_coords["btm_left"],
                        curr_contour_box_coords["btm_right"],
                    ],
                    closest_contour_box_coords["btm_left"][0],
                )
                curr2closest_poi_top = self.get_poi(
                    [
                        curr_contour_box_coords["top_left"],
                        curr_contour_box_coords["top_right"],
                    ],
                    closest_contour_box_coords["top_left"][0],
                )
                closest2curr_poi_top = self.get_poi(
                    [
                        closest_contour_box_coords["top_left"],
                        closest_contour_box_coords["top_right"],
                    ],
                    curr_contour_box_coords["top_right"][0],
                )
                closest2curr_poi_btm = self.get_poi(
                    [
                        closest_contour_box_coords["btm_left"],
                        closest_contour_box_coords["btm_right"],
                    ],
                    curr_contour_box_coords["btm_right"][0],
                )

                if (
                    abs(
                        curr2closest_poi_btm - closest_contour_box_coords["btm_left"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        curr2closest_poi_top - closest_contour_box_coords["top_left"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        closest2curr_poi_top - curr_contour_box_coords["top_right"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        closest2curr_poi_btm - curr_contour_box_coords["btm_right"][1]
                    )
                    < 0.5 * self.common_height
                ):
                    return True
            else:
                curr2closest_poi_btm = self.get_poi(
                    [
                        curr_contour_box_coords["btm_left"],
                        curr_contour_box_coords["btm_right"],
                    ],
                    closest_contour_box_coords["btm_right"][0],
                )
                curr2closest_poi_top = self.get_poi(
                    [
                        curr_contour_box_coords["top_left"],
                        curr_contour_box_coords["top_right"],
                    ],
                    closest_contour_box_coords["top_right"][0],
                )
                closest2curr_poi_top = self.get_poi(
                    [
                        closest_contour_box_coords["top_left"],
                        closest_contour_box_coords["top_right"],
                    ],
                    curr_contour_box_coords["top_left"][0],
                )
                closest2curr_poi_btm = self.get_poi(
                    [
                        closest_contour_box_coords["btm_left"],
                        closest_contour_box_coords["btm_right"],
                    ],
                    curr_contour_box_coords["btm_left"][0],
                )

                if (
                    abs(
                        curr2closest_poi_btm
                        - closest_contour_box_coords["btm_right"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        curr2closest_poi_top
                        - closest_contour_box_coords["top_right"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        closest2curr_poi_top - curr_contour_box_coords["top_left"][1]
                    )
                    < 0.5 * self.common_height
                    or abs(
                        closest2curr_poi_btm - curr_contour_box_coords["btm_left"][1]
                    )
                    < 0.5 * self.common_height
                ):
                    return True
        return False

    def is_new_line(self, curr_contour, curr_line, lines):
        """

        check if the latest contour belongs to the same line
        :param curr_coords:  co-ordinates of the current contour
        :param curr_line: current row
        :param lines:  all rows/lines found
        :return:
        """
        curr_coords = curr_contour["pts"]
        touching_prev_line = False
        if len(lines) > 0:
            prev_line = lines[-1]
            touching_prev_line = self.is_vertically_close(
                curr_coords, prev_line
            )  # check if the contour is close to the last row in lines
        [curr_x, curr_y, curr_w, curr_h] = curr_coords
        if curr_h >= 0.5 * self.common_height:
            closest_contour = self.get_closest_contour(curr_coords, curr_line)
            [prev_x, prev_y, prev_w, prev_h] = closest_contour["pts"]
            min_gap = self.get_min_dist([prev_x, prev_w], [curr_x, curr_w])
            min_gap_percent = min_gap / self.im_shape[1]
            curr_ymax = curr_y + curr_h
            prev_ymax = prev_y + prev_h
            ref_h = self.common_height

            rel_diff_1 = 0.2 + min_gap_percent
            rel_diff_2 = 0.1 + min_gap_percent

            if rel_diff_1 < 0.5:
                rel_diff_1 = 0.5
            # elif rel_diff_1 > 0.75:
            else:
                rel_diff_1 = 0.75

            if rel_diff_2 < 0.3:
                rel_diff_2 = 0.3
            # elif rel_diff_2 > 0.5:
            else:
                rel_diff_2 = 0.5
            overlap_val_x = overlap(curr_x, curr_x + curr_w, prev_x, prev_x + prev_w)
            overlap_val_y = overlap(curr_y, curr_y + curr_h, prev_y, prev_y + prev_h)
            if (
                (not touching_prev_line)
                and (
                    (
                        abs(curr_y - prev_y) > rel_diff_1 * ref_h
                        and abs(curr_ymax - prev_ymax) > rel_diff_2 * ref_h
                    )
                    or (
                        abs(curr_y - prev_y) > rel_diff_2 * ref_h
                        and abs(curr_ymax - prev_ymax) > rel_diff_1 * ref_h
                    )
                )
                and (not self.falls_in_the_area_of_line(curr_contour, closest_contour))
                and (not (overlap_val_x > curr_h and overlap_val_y > 0.7 * curr_h))
            ):
                return True
            # if len(curr_line) > 1 and abs(curr_y-prev_y)>3:
            #     ## check if the sequence of increasing/decreasing x co-ords has changed
            #     [prev2_x, prev2_y, prev2_w, prev2_h] = curr_line[-2]['pts']
            #     if (curr_x - prev_x) * (prev_x-prev2_x)<0:
            #         return True
        if (
            len(curr_line) > 0
            and overlaps_x_coord_any(
                curr_coords, [c["pts"] for c in curr_line], self.common_height
            )
        ) or abs(curr_line[-1]["pts"][1] - curr_y) > 4 * self.common_height:
            return True
        return False

    def compare_overlaps(self, curr_contour, prev_line, curr_line):
        prev_line_closest_contour = self.get_closest_contour(
            curr_contour["pts"], prev_line
        )
        curr_line_closest_contour = self.get_closest_contour(
            curr_contour["pts"], curr_line
        )
        [x, y, w, h] = curr_contour["pts"]
        [p_x, p_y, p_w, p_h] = prev_line_closest_contour["pts"]
        [c_x, c_y, c_w, c_h] = curr_line_closest_contour["pts"]
        prev_y_overlap = overlap(y, y + h, p_y, p_y + p_h)
        curr_y_overlap = overlap(y, y + h, c_y, c_y + c_h)
        if prev_y_overlap > curr_y_overlap:
            return True
        return False

    def check_if_line(self, line):
        max_w = max([r["pts"][2] for r in line])
        if len(line) > 0 or (
            max([l["pts"][2] for l in line]) > 30
            and max([l["pts"][3] for l in line]) > 15
        ):
            return True
        return False

    def sort_contours(self, rois):
        """

        :param rois: extracted rois
        :return: sorted lines top to bottom and words left to right in each line
        """
        rois = sorted(rois, key=lambda x: x["pts"][1])
        lines = []
        line = []
        if len(rois) > 1:
            line.append(rois[0])
            curr_ymin = rois[0]["pts"][1]
            curr_h = rois[0]["pts"][3]
            curr_ymax = curr_ymin + curr_h
            for c in rois[1:]:
                [xmin, ymin, width, height] = c["pts"]
                ymax = ymin + height
                if len(lines) > 0 and (
                    self.is_new_line(c, lines[-1], lines[:-1]) is False
                    and self.compare_overlaps(c, lines[-1], line) is True
                ):  # check if it's a part of the previous line
                    lines[-1].append(c)  # add contour to prev line
                elif self.is_new_line(c, line, lines):
                    if len(line) > 0 and self.check_if_line(line):
                        lines.append(line)
                    line = []
                    line.append(c)
                else:
                    # (height < 0.6*curr_h and (abs(ymin - curr_ymin) < 0.75 * curr_h) ): ## to include comma in the same line
                    line.append(c)
            if len(line) > 0 and self.check_if_line(line):
                lines.append(line)
            # sort contours inside the line
        for idx, line in enumerate(lines):
            sorted_line = sorted(line, key=lambda x: x["pts"][0])
            lines[idx] = sorted_line
            # print("IDS -",[s['id'] for s in sorted_line])
        return lines

    def get_common_height(self, rois):
        heights = []
        print("INSIDE COMMON HEIGHT")
        for c in rois:
            [x, y, w, h] = c["pts"]
            if c["pts"][3] > 10 and (w > 2 * h or h > 2 * w):
                heights.append(c["pts"][3])
        if len(heights) > 0:
            return np.bincount(heights).argmax()

        else:
            return 0

    # def get_common_height(self):
    #     heights = []
    #     for line in self.lines:
    #         for word in line:
    #             heights.append(word["pts"][3])
    #     if len(heights)>0:
    #         return np.bincount(heights).argmax()
    #     else:
    #         return 0

    def resize_image(self, img):
        final_width = 2000
        fx = final_width / img.shape[1]
        fy = fx
        if final_width < img.shape[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)
        print("final size", resized.shape)
        return resized

    def remove_bg(self, image, blur_window=51):
        [h, w] = image.shape[:2]
        if blur_window != 51:
            blur_window = int(min(h / 4, w / 4))
            if blur_window % 2 == 0:
                blur_window += 1
        blur_bg = cv2.medianBlur(image, blur_window)

        if self.debug:
            cv2.imwrite("debug/blur_bg.jpg", np.bitwise_not(blur_bg))
        # wo_bg = cv2.subtract(np.bitwise_not(image), np.bitwise_not(blur_bg))
        wo_bg = np.subtract(
            np.bitwise_not(image).astype(int), np.bitwise_not(blur_bg).astype(int)
        )
        wo_bg[wo_bg < -50] = 255
        wo_bg[wo_bg < 0] = 0
        wo_bg = wo_bg.astype(np.uint8)
        if self.debug:
            cv2.imwrite("debug/wo_bg.jpg", wo_bg)
        morphKernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        wo_bg = cv2.morphologyEx(wo_bg, cv2.MORPH_CLOSE, morphKernel)
        return wo_bg

    def replace_indices(self, a, val_old, val_new):
        arr = np.empty(a.max() + 1, dtype=val_new.dtype)
        arr[val_old] = val_new
        return arr[a]

    def remove_snp(self, img):
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, None, None, None, 8, cv2.CV_32S
        )
        sizes = stats[1:, -1]  # get CC_STAT_AREA component
        img2 = np.zeros((labels.shape), np.uint8)
        rel_mask_indices = [n for n in range(0, nlabels - 1) if sizes[n] >= 3]
        irr_mask_indices = [n for n in range(0, nlabels - 1) if sizes[n] < 3]

        new_arr = np.array([-1] * len(irr_mask_indices))
        # labels[irr_mask_indices] = new_arr
        if len(irr_mask_indices) > 0:
            labels = self.replace_indices(labels, np.array(irr_mask_indices), new_arr)
            img2[labels == -1] = 255

        img2 = cv2.bitwise_not(img2)
        """
        if len(irr_mask_indices)<len(rel_mask_indices):
            for i in irr_mask_indices:
                img2[labels == i + 1] = 255
            img2 = cv2.bitwise_not(img2)
        else:
            for i in rel_mask_indices:
                img2[labels == i + 1] = 255
        """

        # for i in range(0, nlabels - 1):
        #     if sizes[i] >= 3:  # filter small dotted regions
        #         img2[labels == i + 1] = 255
        #     else:
        #         c = 2
        # img = cv2.bitwise_not(img)
        img = cv2.bitwise_and(img, img2)
        median_blur = cv2.medianBlur(img, 3)
        return cv2.bitwise_and(median_blur, img)

    def threshold(self, image, basename):
        [or_h, or_w] = image.shape[:2]
        image = cv2.resize(image, None, None, fx=0.5, fy=0.5)
        width = image.shape[1]
        height = image.shape[0]
        adap_thresh_window = min(int(width / 10), int(height / 10))
        if adap_thresh_window % 2 == 0:
            adap_thresh_window += 1
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        st_t = time.time()
        gray_wo_bg = self.remove_bg(gray, 51)  # remove any background color
        blur = cv2.medianBlur(gray_wo_bg, 3)  # to remove background noise
        # print('Time taken for noise rem',time.time()-st_t)
        morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, morphKernel)
        # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        # grad_dil = cv2.dilate(grad,dilate_kernel)
        if self.debug:
            cv2.imwrite("debug/gradient.jpg" + basename, grad)
        st_t = time.time()
        # print("adap win",adap_thresh_window,"height",height,"width",width)
        adap_thresh = cv2.adaptiveThreshold(
            gray_wo_bg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -2
        )
        cv2.imwrite("gray_wo.jpg", gray_wo_bg)
        cv2.imwrite("adap.jpg", adap_thresh)
        # print('Time taken for adap thresh',time.time()-st_t)
        adap_thresh = cv2.resize(adap_thresh, (or_w, or_h))
        gray_wo_bg = cv2.resize(gray_wo_bg, (or_w, or_h))
        # print('adap shape',adap_thresh.shape)
        # print('gray_wo_bg',gray_wo_bg.shape)
        return adap_thresh, gray_wo_bg

    def getLines(self, thresh_img, h, w):
        # cv2.imwrite("thresh.jpg",thresh_img)
        horizontal = np.copy(thresh_img)
        vertical = np.copy(thresh_img)
        rows = h
        cols = w
        verticalsize = rows // 50
        horSize = cols // 10
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        horStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horSize, 1))

        verticalStructure_dil = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, verticalsize)
        )
        horStructure_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (horSize, 3))

        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure_dil, iterations=3)

        hor = cv2.erode(horizontal, horStructure)
        hor = cv2.dilate(hor, horStructure_dil, iterations=3)
        lines = cv2.bitwise_or(hor, vertical)
        return hor, vertical, lines

    def getVerticalLines(self):
        # print("PATH--", path)
        # im = cv2.imread(self.modified_path)
        im = self.oriented_orig_img.copy()
        if len(im.shape) == 3:
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            gray = im
        gray = cv2.bitwise_not(gray)
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        vertical = np.copy(bw)
        rows = vertical.shape[0]
        verticalsize = rows // 50
        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        contours, hierarchy = cv2.findContours(
            vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        vertical_lines = []
        for idx, c in enumerate(contours):
            sort_c = sorted(c, key=lambda x: x[0][1])
            pt1 = tuple(sort_c[0][0])
            pt2 = tuple(sort_c[-1][0])
            vertical_lines.append([pt1, pt2])
        return vertical_lines

    def get_roi(self, img_thresh, unet_oriented):
        t1 = time.time()
        canvas = self.oriented_orig_img.copy()
        [im_h, im_w] = img_thresh.shape[:2]
        # remove all lines
        # bw = cv2.bitwise_and(img_thresh, cv2.bitwise_not(lines))
        # cv2.imwrite('img_thresh.jpg',img_thresh)

        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        unet_oriented = cv2.morphologyEx(unet_oriented, cv2.MORPH_CLOSE, rectKernel)
        bw = cv2.bitwise_and(img_thresh, unet_oriented)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        gradX = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if self.debug:
            cv2.imwrite("filters/thresh_" + self.basename, thresh)
        # print('Time for preprocessing',time.time()-t1)
        cv2.imwrite("thresh.jpg", cv2.bitwise_and(thresh, unet_oriented))
        cnts = cv2.findContours(
            cv2.bitwise_and(thresh, unet_oriented),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[-2]
        # print('Time after find contours',time.time()-t1)
        # cnts = cv2.findContours(unet_oriented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        rois = []
        mask = np.zeros(self.oriented_orig_img.shape[:2], np.uint8)
        for idx, c in enumerate(cnts):
            [x, y, w, h] = cv2.boundingRect(c)
            # define main island contour approx. and hull
            perimeter = cv2.arcLength(c, True)
            epsilon = 0.05 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(c) > 5 and (h > 10 or (h > 5 and w * h > 50)) and w > 0:
                # cv2.drawContours(canvas,[c],-1,(0,0,255),2)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255), 2)
                self.contours_info[self.current_contour_idx] = c
                """
                x2 = min(im_w, x + w + 5)
                x = max(x-5,0)
                y2 = min(im_h, y+h+5)
                y = max(y-5,0)
                """
                rois.append({"id": self.current_contour_idx, "pts": [x, y, w, h]})
                self.current_contour_idx += 1
            else:
                cv2.drawContours(mask, [c], -1, 255, -1)
        # mask = np.zeros(self.oriented_orig_img.shape[:2], np.uint8)
        # cv2.drawContours(mask, cnts,-1, 255, -1)
        # print('END get_roi time ',time.time()-t1)
        return rois

    def split_text_by_vertical_lines(self):
        fin_rois = []
        for region in self.regions:
            [x1, y1, w, h] = region["pts"]
            y_mid = int(y1 + (h / 2))
            intersecting_pts = is_separated_by_vertical_line(
                [[(x1, y1), (x1 + w, y1)], [(x1, y1 + h), (x1 + w, y1 + h)]],
                self.verticalLines,
                multiple=True,
            )
            # intersecting_pt_bottom = is_separated_by_vertical_line(, self.verticalLines)
            if intersecting_pts and len(intersecting_pts) == 1:
                inter_x = int(intersecting_pts[0].x)
                if inter_x - 1 - x1 > 0:
                    fin_rois.append(
                        {
                            "id": str(self.current_contour_idx),
                            "pts": [x1, y1, inter_x - 1 - x1, h],
                        }
                    )
                    self.contours_info[
                        str(self.current_contour_idx)
                    ] = self.contours_info[region["id"]]
                    self.current_contour_idx += 1
                if int(w - (inter_x + 2 - x1)) > 0:
                    fin_rois.append(
                        {
                            "id": str(self.current_contour_idx),
                            "pts": [inter_x + 2, y1, int(w - (inter_x + 2 - x1)), h],
                        }
                    )
                    self.contours_info[
                        str(self.current_contour_idx)
                    ] = self.contours_info[region["id"]]
                    self.current_contour_idx += 1
            elif intersecting_pts:
                prev_x = x1
                for pt in intersecting_pts:
                    inter_x = int(pt.x)
                    if inter_x - 1 - prev_x > 0:
                        fin_rois.append(
                            {
                                "id": str(self.current_contour_idx),
                                "pts": [prev_x, y1, inter_x - 1 - prev_x, h],
                            }
                        )
                        self.contours_info[
                            str(self.current_contour_idx)
                        ] = self.contours_info[region["id"]]
                        self.current_contour_idx += 1
                        prev_x = inter_x + 2
                if int(w - (prev_x - x1)) > 0:
                    fin_rois.append(
                        {
                            "id": str(self.current_contour_idx),
                            "pts": [prev_x, y1, int(w - (prev_x - x1)), h],
                        }
                    )
                    self.contours_info[
                        str(self.current_contour_idx)
                    ] = self.contours_info[region["id"]]
                    self.current_contour_idx += 1
            else:
                fin_rois.append(region)
        return fin_rois

    def get_errored_contours(self, line_idx):
        line = self.lines[line_idx]
        perfect_line = False
        to_verify = line
        to_verify = sorted(to_verify, key=lambda x: x["pts"][3])  # sort by height
        res = []
        max_line_height = max([l["pts"][3] for l in line])
        while (
            not perfect_line
            and len(to_verify) > 0
            and not (
                (
                    len(line) > 1
                    and contains_same_heights_2([t["pts"] for t in to_verify])
                )
                or (len(line) == 1 and (max_line_height < 1.9 * self.common_height))
            )
        ):
            res.append(to_verify[-1])
            to_verify = to_verify[:-1]
        return res, to_verify

    # def is_steep(self,data,trough_idx):
    #     left_vals = [data[trough_idx-1],data[trough_idx-2],data[trough_idx-3],data[trough_idx-4],data[trough_idx-5]]
    #     right_vals = [data[trough_idx+1],data[trough_idx+2],data[trough_idx+3],data[trough_idx+4],data[trough_idx+5]]
    #     curr_val = data[trough_idx]
    #     left_diff = [abs(v-curr_val) for v in left_vals]
    #     right_diff = [abs(v-curr_val) for v in right_vals]
    #     if max(left_diff)>35 :
    #         return True
    #     return False

    def is_steep(self, data, trough_idx):
        peak_val = data[trough_idx]
        if peak_val > -400:
            break_point = 35
        else:
            break_point = 150
        for idx in reversed(range(0, trough_idx)):
            curr_val = data[idx]
            if peak_val - curr_val > break_point:
                return True
            elif curr_val > peak_val:
                break

        if peak_val > -400:
            break_point = 100
        else:
            break_point = 200
        for idx in range(trough_idx, len(data)):
            curr_val = data[idx]
            if peak_val - curr_val > break_point:
                return True
            elif curr_val > peak_val:
                break

        return False

    def is_relevant_minima(self, data, trough_idx):
        height = len(data)
        if height > 0.5 * self.common_height:
            if height - 10 > trough_idx > 10 and self.is_steep(data, trough_idx):
                return True
        return False

    def get_contours(self, thresh, canvas, x_start, y_start):
        [height, width] = thresh.shape[:2]
        if width > self.common_height * 2:
            rectKernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (int(self.common_height / 2), 1)
            )
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
        hor_projection = getHorizontalProjectionProfile(thresh)
        hor_projection = np.array([-1 * p for p in hor_projection])
        smooth_data = smooth(hor_projection)
        troughs = argrelmax(smooth_data, order=3)
        split_points = [0]
        res_contours = []
        for t in troughs[0]:
            if self.is_relevant_minima(smooth_data, t):
                split_points.append(t)
                if self.debug:
                    plt.plot(t, smooth_data[t], "g*")
        if self.debug:
            plt.plot(smooth_data)
            plt.show()
        for idx, y_coord in enumerate(split_points):
            if y_coord == 0:
                ymin = y_coord
            else:
                ymin = y_coord - 2
            if idx == len(split_points) - 1:
                y_max = height
            else:
                y_max = split_points[idx + 1] + 2
            if y_max - ymin > 15:
                if self.debug:
                    cv2.rectangle(canvas, (0, ymin), (width, y_max), (0, 255, 0), 3)
                snip_thresh = thresh[ymin:y_max, :]
                cnts = cv2.findContours(
                    snip_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[-2]
                for c in cnts:
                    [x, y, w, h] = cv2.boundingRect(c)
                    self.current_contour_idx += 1
                    res_contours.append(
                        {
                            "id": self.current_contour_idx,
                            "pts": [x_start + x, y_start + ymin + y, w, h],
                        }
                    )

        if self.debug:
            cv2.imshow("w2", canvas)
            cv2.waitKey(0)
        return res_contours

    def split_text_lines(self, errored_contours):
        thresh_img = self.oriented_img_thresh
        res_contours = []
        for c in errored_contours:
            [x, y, w, h] = c["pts"]
            deb = self.oriented_orig_img[y : y + h, x : x + w]
            if True:
                # if h>1.2*self.common_height:
                canvas = self.oriented_orig_img.copy()
                mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
                contour = self.contours_info[c["id"]]
                cv2.drawContours(mask, [contour], -1, 255, -1)
                canvas = np.bitwise_and(thresh_img, mask)
                bw = cv2.bitwise_and(canvas, self.text_mask)

                rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
                gradX = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, rectKernel)

                thresh = cv2.threshold(
                    gradX, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )[1]
                res_contours.extend(
                    self.get_contours(
                        thresh[y : y + h, x : x + w],
                        self.oriented_orig_img.copy()[y : y + h, x : x + w, :],
                        x,
                        y,
                    )
                )
            else:
                res_contours.append(c)
        return res_contours

    def extract_roi_using_sp(self, line_indices, snip_thresh_dilated, snip_lines_img):
        # hor_projection = getHorizontalProjectionProfile(
        #     cv2.bitwise_and(snip_thresh_dilated, cv2.bitwise_not(snip_lines_img)))
        # hor_projection = np.array([-1 * p for p in hor_projection])
        # smooth_data = smooth(hor_projection)
        # # troughs = find_peaks(hor_projection)
        # window = signal.general_gaussian(5, p=0.5, sig=10)
        # filtered = signal.fftconvolve(window, hor_projection)
        # filtered = (np.average(hor_projection) / np.average(filtered)) * filtered
        # # smooth_data = np.roll(filtered, -25)
        # troughs_2 = argrelmax(smooth_data, order=3)
        # # plt.plot(hor_projection)
        # # smooth_data = pd.Series(hor_projection).rolling(window=5).mean().plot(style='k')
        # # smooth_data = savgol_filter(hor_projection, 11, 3)
        # # plt.plot(smooth_data)
        # # for t in troughs_2[0]:
        # #     if smooth_data[t] > -400:
        # #         plt.plot(t,smooth_data[t],"g*")
        # # plt.show()
        # # return False
        rois = []
        for idx in line_indices:
            line = [l["pts"] for l in self.lines[idx]]
            max_line_height = max([l[3] for l in line])
            if (len(line) > 1 and contains_same_heights(line)) or (
                len(line) == 1 and (max_line_height < 1.9 * self.common_height)
            ):
                rois.extend(self.lines[idx])
            else:
                errored_contours, correct_contours = self.get_errored_contours(idx)
                rois.extend(correct_contours)
                rois.extend(self.split_text_lines(errored_contours))
        return rois

    def extract_cropped_text(self, y_min, y_max, x_min, x_max, line_indices_to_modify):
        snippet_color = self.oriented_orig_img[y_min:y_max, x_min:x_max]
        cp = snippet_color.copy()
        cv2.imshow("w", cp)
        cv2.waitKey(0)
        snippet = self.oriented_wo_bg[y_min:y_max, x_min:x_max]
        snip_thresh = cv2.threshold(
            snippet, 80, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
        )[1]
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        snip_dilated = cv2.morphologyEx(snip_thresh, cv2.MORPH_CLOSE, rectKernel)
        snip_thresh_dilated = cv2.threshold(
            snip_dilated, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        # snip_thresh_dilated = cv2.bitwise_and(snip_thresh_dilated,self.text_mask[y_min:y_max, x_min:x_max])
        snip_thresh_dilated = self.text_mask[y_min:y_max, x_min:x_max]
        snip_thresh_dilated = cv2.bitwise_and(snip_thresh_dilated, snip_dilated)
        snip_thresh_dilated = cv2.morphologyEx(
            snip_thresh_dilated, cv2.MORPH_CLOSE, rectKernel
        )

        # for idx in line_indices_to_modify:
        #     for word in self.lines[idx]:
        #         [x,y,w,h] = word["pts"]
        #         word_snip = self.oriented_orig_img[y:y+h, x:x+w]
        #         word_snip = cv2.cvtColor(word_snip, cv2.COLOR_BGR2GRAY)
        #         word_snip_thresh = cv2.threshold(word_snip, 80, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
        #         word_snip_dilated = cv2.morphologyEx(word_snip_thresh, cv2.MORPH_CLOSE, rectKernel)
        #         word_snip_thresh_dilated = cv2.threshold(word_snip_dilated, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #         word_snip_thresh_dilated = cv2.bitwise_and(word_snip_thresh_dilated,
        #                                               cv2.bitwise_not(self.all_lines_img[y:y+h, x:x+w]))
        #         word_hor_projection = getHorizontalProjectionProfile(
        #             cv2.bitwise_and(word_snip_thresh_dilated, cv2.bitwise_not(self.all_lines_img[y:y+h, x:x+w])))
        #         word_hor_projection = [-1 * p for p in word_hor_projection]
        #         cv2.imshow("w",word_snip_thresh)
        #         cv2.waitKey(0)
        #         smooth_data = smooth(word_hor_projection)
        #         troughs_2 = argrelmax(smooth_data, order=3)
        #         # plt.plot(hor_projection)
        #         # smooth_data = pd.Series(hor_projection).rolling(window=5).mean().plot(style='k')
        #         # smooth_data = savgol_filter(hor_projection, 11, 3)
        #         plt.plot(smooth_data)
        #         for t in troughs_2[0]:
        #             if smooth_data[t] > -400:
        #                 plt.plot(t, smooth_data[t], "g*")
        #         plt.show()

        contours = cv2.findContours(
            snip_thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[-2]
        rois = []
        for idx, c in enumerate(contours):
            [x, y, w, h] = cv2.boundingRect(c)
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(c) > 5 and h > 10:
                c_id = "c_" + str(self.current_contour_idx)
                rois.append({"id": c_id, "pts": [x, y, w, h]})
                self.contours_info[c_id] = c
                self.current_contour_idx += 1
                cv2.drawContours(cp, c, -1, (0, 0, 255), -1)
                # cv2.rectangle(cp,(x,y),(x+w,y+h),(0,0,255),3)
        roi_lines = self.sort_contours(rois)
        is_perfect = True
        for line in roi_lines:
            line = [l["pts"] for l in line]
            max_line_height = max(l[3] for l in line)
            if (not contains_same_heights(line)) or (
                len(line) == 1 and (max_line_height > 1.9 * self.common_height)
            ):
                is_perfect = False
                break
        if is_perfect:
            return transform_coords(roi_lines, y_min, x_min)
        else:
            rois = self.extract_roi_using_sp(
                line_indices_to_modify,
                snip_thresh_dilated,
                self.text_mask[y_min:y_max, x_min:x_max],
            )
            return self.sort_contours(rois)

    def remove_noise(self):
        fin_lines = []
        for line_idx, line in enumerate(self.lines):
            max_w = max([l["pts"][2] for l in line])
            max_h = max([l["pts"][3] for l in line])
            if len(line) > 1:
                min_w = 2 * self.common_height
            else:
                min_w = 2 * self.common_height
            if (
                max_w > min_w and max_h > 0.5 * self.common_height
            ) or max_h >= 0.75 * self.common_height:
                fin_lines.append(line)
        return fin_lines

    def verifyLines(self):
        fin_lines = []
        modified_indices = []
        for line_idx, line in enumerate(self.lines):
            line = [l["pts"] for l in line]
            if line_idx not in modified_indices:
                max_line_height = max([r[3] for r in line])
                if (not contains_same_heights(line)) or (
                    max_line_height > 1.9 * self.common_height
                ):
                    y_max = max([b[1] + b[3] for b in line])
                    line_indices_to_modify = []
                    for idx in range(line_idx, len(self.lines)):
                        curr_line = self.lines[idx]
                        curr_line = [c["pts"] for c in curr_line]
                        curr_max_line_height = max([r[3] for r in curr_line])
                        if contains_same_heights(curr_line) and (
                            curr_max_line_height <= 1.9 * self.common_height
                        ):
                            curr_y2_max = max([b[1] + b[3] for b in curr_line])
                            curr_y2_min = min([b[1] + b[3] for b in curr_line])
                            if curr_y2_max <= y_max:
                                line_indices_to_modify.append(idx)
                            # elif y_max-curr_y2_min>-2:
                            #     line_indices_to_modify.append(idx)
                            #     y_max = curr_y2_max
                            else:
                                break
                        else:
                            line_indices_to_modify.append(idx)
                            y_max = max([b[1] + b[3] for b in curr_line])
                    if len(line_indices_to_modify) > 0:
                        x_min = line[0][0]
                        x_max = line[0][0] + line[0][2]
                        y_min = line[0][1]
                        y_max = line[0][1] + line[0][3]
                        for idx in line_indices_to_modify:
                            curr_line = self.lines[idx]
                            curr_line = [l["pts"] for l in curr_line]
                            line_x_min = min(b[0] for b in curr_line)
                            line_x_max = max(b[0] + b[2] for b in curr_line)
                            line_y_min = min(b[1] for b in curr_line)
                            line_y_max = max(b[1] + b[3] for b in curr_line)
                            if line_x_min < x_min:
                                x_min = line_x_min
                            if line_x_max > x_max:
                                x_max = line_x_max
                            if line_y_min < y_min:
                                y_min = line_y_min
                            if line_y_max > y_max:
                                y_max = line_y_max
                        # cv2.imshow("w",self.oriented_orig_img[y_min:y_max,x_min:x_max])
                        # cv2.waitKey(0)
                        m_t = time.time()
                        modified_roi_lines = self.extract_cropped_text(
                            y_min, y_max, x_min, x_max, line_indices_to_modify
                        )
                        # print('Time taken for extract_cropped_text',time.time()-m_t)
                        if modified_roi_lines:
                            fin_lines.extend(modified_roi_lines)
                            modified_indices.extend(line_indices_to_modify)
                        else:
                            fin_lines.append(self.lines[line_idx])
                else:
                    fin_lines.append(self.lines[line_idx])
        return fin_lines

    def checkIfVerLinePasses(self, sent_obj):
        [x1, y1, x2, y2] = sent_obj["pts"]
        y_mid = int(y1 + ((y2 - y1) / 2))
        intersecting_pt = is_separated_by_vertical_line(
            [(x1, y_mid), (x2, y_mid)], self.verticalLines
        )
        if intersecting_pt:
            xMaxArr = []
            xMinArr = []
            yMinArr = []
            yMaxArr = []
            idArr = []
            word_ids = sent_obj["ids"]
            for id in word_ids:
                wrd = self.words[id]
                for char_text, char_vertices in zip(wrd["texts"], wrd["vertices"]):
                    xAll = list()
                    yAll = list()
                    for vertex in char_vertices:
                        xAll.append(vertex["x"])
                        yAll.append(vertex["y"])
                    xMaxArr.append(max(xAll))
                    xMinArr.append(min(xAll))
                    yMinArr.append(min(yAll))
                    yMaxArr.append(max(yAll))
                    idArr.append(id)
            # print(xMaxArr)
            # print(xMinArr)
            char_idx = 0
            char_xmins = list()
            char_xmaxs = list()
            char_ymins = list()
            char_ymaxs = list()
            for char in list(sent_obj["text"]):
                if char != " ":
                    char_xmins.append(xMinArr[char_idx])
                    char_xmaxs.append(xMaxArr[char_idx])
                    char_ymins.append(yMinArr[char_idx])
                    char_ymaxs.append(yMaxArr[char_idx])
                    char_idx += 1
                else:
                    char_xmins.append(None)
                    char_xmaxs.append(None)
                    char_ymins.append(None)
                    char_ymaxs.append(None)
            intersecting_x = intersecting_pt.x
            split_idx = 0
            for idx, char_xmax in enumerate(char_xmaxs):
                if char_xmax and char_xmax > intersecting_x:
                    split_idx = idx
                    break
            if split_idx > 0:
                res = []
                chars_list = list(sent_obj["text"])
                prev_idx = None
                for i in range(split_idx - 1, -1, -1):
                    if char_xmaxs[i]:
                        prev_idx = i
                        break
                if prev_idx:
                    res.append(
                        {
                            "text": "".join(chars_list[:split_idx]),
                            "pts": [x1, y1, char_xmaxs[prev_idx], char_ymaxs[prev_idx]],
                            "ids": list(set(idArr[:split_idx])),
                        }
                    )
                    res.append(
                        {
                            "text": "".join(chars_list[split_idx:]),
                            "pts": [
                                char_xmins[split_idx],
                                char_ymins[split_idx],
                                x2,
                                y2,
                            ],
                            "ids": list(set(idArr[split_idx:])),
                        }
                    )
                    return res
                else:
                    return [sent_obj]
            else:
                return [sent_obj]
        else:
            return [sent_obj]

    def add_padding(self):
        [im_h, im_w] = self.im_shape[:2]

        for line_idx, line in enumerate(self.lines):
            for block_idx, block in enumerate(line):
                [x, y, w, h] = block["pts"]
                # print(x,y,w,h, "im_w",im_w,"im_h",im_h)
                # x2 = min(im_w, x + w + 2)
                # x = max(x - 2, 0)
                x2 = x + w
                y2 = min(im_h, y + h + 2)
                y = max(y - 2, 0)
                self.lines[line_idx][block_idx]["pts"] = [x, y, x2 - x, y2 - y]

    def merge_close_texts(self):
        im_width = self.im_shape[1]
        if im_width < 1500:
            max_inter_word_spacing = 20
            min_inter_word_spacing = 2
            min_word_height = 5
        else:
            max_inter_word_spacing = int(15 * (im_width / 1500))
            min_inter_word_spacing = int(2 * (im_width / 1500))
            min_word_height = int(5 * (im_width / 1500))
        fin_lines = []

        # merge blocks if the gap is less than max_inter_word_spacing
        for l in self.lines:
            line_words = l
            # line_words = [word for word in line_words if (word['pts'][3])>min_word_height and len(word['text'])>0]
            # if sum([c.isalpha() for c in l['text']]) > 0 or sum([c.isdigit() for c in l['text']]) > 0:
            if len(line_words) > 1:
                split_indices = []
                prev_x_max = line_words[0]["pts"][0] + line_words[0]["pts"][2]
                prev_char_height = line_words[0]["pts"][3]
                prev_coords = line_words[0]["pts"]
                for idx, block in enumerate(line_words[1:]):
                    [x1, y1, w, h] = block["pts"]
                    x2 = x1 + w
                    y2 = y1 + h
                    curr_char_height = h
                    max_inter_word_spacing = int(
                        1.75 * max(prev_char_height, curr_char_height)
                    )  ## calculate the spacing allowed based on the character width
                    max_inter_word_spacing_2 = int(
                        0.75 * max(prev_char_height, curr_char_height)
                    )  ## calculate the spacing allowed based on the character width
                    max_inter_word_spacing = -1
                    max_inter_word_spacing_2 = -1
                    curr_x_min = x1
                    curr_y_mid = int(y1 + ((y2 - y1) / 2))
                    spacing = curr_x_min - prev_x_max
                    # if "Survey Nos" in l["text"]:
                    #     print(spacing)
                    # save the indices where distance is large enough to be considered as separate blocks
                    # and merge the rest

                    # the contours split using vertical lines share one side and the length of connecting line will be 0
                    # so add some more length
                    if prev_x_max + 3 == curr_x_min:
                        connecting_line = [
                            [min(curr_x_min + 2, im_width), curr_y_mid],
                            [max(0, prev_x_max - 2), curr_y_mid],
                        ]
                    else:
                        connecting_line = [
                            [curr_x_min, curr_y_mid],
                            [prev_x_max, curr_y_mid],
                        ]
                    prev_y1 = prev_coords[1]
                    prev_y2 = prev_coords[1] + prev_coords[3]

                    y_overlap = overlap(prev_y1, prev_y2, y1, y2)

                    if (
                        spacing > max_inter_word_spacing
                        or y_overlap / min(h, prev_char_height) < 0.8
                        or (
                            spacing > max_inter_word_spacing_2
                            and y_overlap / min(h, prev_char_height) < 0.9
                        )
                    ):
                        split_indices.append(idx + 1)
                    # elif spacing > min_inter_word_spacing:
                    #     if is_separated_by_vertical_line([(prev_x_max, curr_y_mid), (curr_x_min, curr_y_mid)],
                    #                                      self.verticalLines):
                    #         split_indices.append(idx + 1)
                    prev_x_max = x2
                    prev_char_height = curr_char_height
                    prev_coords = block["pts"]
                if len(split_indices) > 0:
                    prev_idx = 0
                    line_blocks = []
                    for idx in split_indices:
                        combined = combine(line_words[prev_idx:idx])
                        line_blocks.append(combined)
                        prev_idx = idx
                    combined = combine(line_words[prev_idx:])
                    line_blocks.append(combined)
                    fin_lines.append(line_blocks)
                else:
                    fin_lines.append([combine(l)])
            elif len(line_words) == 1:
                fin_lines.append([combine(line_words[0:1])])
        rois = []
        for line in fin_lines:
            for block in line:
                rois.append(block)
        return self.sort_contours(rois)

    def process_coords(self):
        fin_lines = []
        # merge blocks if the gap is less than max_inter_word_spacing
        for l in self.lines:
            line = []
            for idx, block in enumerate(l):
                [x1, y1, w, h] = block["pts"]
                x2 = x1 + w
                y2 = y1 + h
                block["pts"] = [x1, y1, x2, y2]
                line.append(block)
            fin_lines.append(line)
        return fin_lines
