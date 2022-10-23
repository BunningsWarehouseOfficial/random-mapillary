"""Divide the combined real label mask for a sign into quadrants.

Code adapted from: https://stackoverflow.com/a/60314364/12350950
"""

import cv2
import numpy as np
from skimage import io              # Only needed for web reading images
from skimage.measure import regionprops


def read_image(img_url):
    """Read image from URL or local path."""
    if img_url.startswith("http"):
        img = cv2.cvtColor(io.imread(img_url), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(cv2.imread(img_url), cv2.COLOR_RGB2BGR)

    # Inverse binary threshold grayscale version of image
    # Assumption: plain white background
    return cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 248, 255, cv2.THRESH_BINARY_INV)[1]

def divide_sign_mask_quadrants(img_thr, filename, debug=False, save=True):
    """Divide the combined real label mask for a sign into quadrants.
    Mask must be a binary greyscale mask with a black background.

    `filename` arg should include file extension.
    """
    height, width = img_thr.shape[:2]
    img = img_thr.copy()
    img_c = img_thr.copy()

    # Find external contour for minimum area rectangle
    # Assumption: only one object/contour
    cnts = cv2.findContours(img_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Find rotated rectangle of the minimum area
    min_rect = cv2.minAreaRect(cnts[0])  # ((tl_x, tl_y), (w, h), angle)
    rect_pts = np.int0(cv2.boxPoints(min_rect))  # tl, tr, bl, br
    pre_alpha = min_rect[2]
    alpha = pre_alpha
    if np.abs(alpha) > 45:
        alpha = alpha + 90

     # Find standard bounding box rectangle
    rect = cv2.boundingRect(cnts[0])

    # Determine whether to use a bounding rectangle or minimum area rectangle
    pixel_area = regionprops(img_thr)[0].area
    minbox_area = min_rect[1][0] * min_rect[1][1]
    minbox_coverage = pixel_area / minbox_area
    w = min_rect[1][0]
    h = min_rect[1][1]
    edge_ratio = max(w, h) / min(w, h)  # 1 if it is a square
    using_minbox = (minbox_coverage > 0.9 and
                       # Exclude diamond-shaped signs if they are sufficiently square
                       (pre_alpha < 35 or pre_alpha > 55 or edge_ratio > 1.2))
    if using_minbox:
        # Calculate centroid
        cent = np.int0((rect_pts[0] + rect_pts[2]) / 2)

        # Calculate tangent of rotation angle
        tan_alpha = np.tan(np.deg2rad(alpha))

        c_x, c_y = cent[0], cent[1]

        # Calculate first edge point
        x0 = np.int32(c_x - c_y / tan_alpha)
        pre_x0 = x0
        if alpha == 180 or alpha == 0:
            x0 = 0
            y0 = c_y
        elif x0 < 0:
            x0 = 0
            y0 = np.int32(-c_x * tan_alpha + c_y)
        elif x0 > width:
            x0 = width
            y0 = np.int32((width - c_x) * tan_alpha + c_y)
        else:
            y0 = 0

        # Calculate second edge point
        x1 = np.int32(c_x + (height - c_y) / tan_alpha)
        pre_x1 = x1
        if alpha == 180 or alpha == 0:
            x1 = width
            y1 = c_y
        elif x1 > width:
            x1 = width
            y1 = np.int32((x1 - c_x) * tan_alpha + c_y)
        elif x1 < 0:
            x1 = 0
            y1 = np.int32(-c_x * tan_alpha + c_y)
        else:
            y1 = height

        # Calculate third edge point
        tan_alpha = np.tan(np.deg2rad(alpha + 90))
        x2 = np.int32(c_x - c_y / tan_alpha)
        if x2 < 0:
            x2 = 0
            y2 = np.int32(-c_x * tan_alpha + c_y)
        else:
            y2 = 0

        # Calculate fourth edge point
        x3 = np.int32(c_x + (height - c_y) / tan_alpha)
        if x3 > width:
            x3 = width
            y3 = np.int32((x3 - c_x) * tan_alpha + c_y)
        else:
            y3 = height
    else:
        # Use bounding rectangle
        cent = (np.int0(rect[0] + rect[2] / 2), np.int0(rect[1] + rect[3] / 2))
        x0, y0 = cent[0], 0
        x1, y1 = cent[0], height
        x2, y2 = 0, cent[1]
        x3, y3 = width, cent[1]

    # Generate mask for horizontal cutting
    # Assumption: Image is sufficiently large
    mask = np.zeros_like(img_thr)
    if x0 > x2 and y0 < y2:  # Make sure horizontal is actually horizontal
        mask = cv2.line(mask, (x0, y0), (x1, y1), 255, 1)
    else:
        mask = cv2.line(mask, (x2, y2), (x3, y3), 255, 1)
    mask_orig = mask.copy()
    cv2.floodFill(mask, None, (cent[0] - 5, cent[1] - 5), 255)
    # Repeat with slightly different seed point if image is unchanged
    if cv2.countNonZero(cv2.subtract(mask, mask_orig)) == 0:
        cv2.floodFill(mask, None, (cent[0] - 7, cent[1] - 3), 255)
        if cv2.countNonZero(cv2.subtract(mask, mask_orig)) == 0:
            cv2.floodFill(mask, None, (cent[0] - 3, cent[1] - 7), 255)

    # Generate mask for vertical cutting
    mask_v = np.zeros_like(img_thr)
    if x0 > x2 and y0 < y2:  # Make sure vertical is actually vertical
        mask_v = cv2.line(mask_v, (x2, y2), (x3, y3), 255, 1)
    else:
        mask_v = cv2.line(mask_v, (x0, y0), (x1, y1), 255, 1)
    mask_v_orig = mask_v.copy()
    cv2.floodFill(mask_v, None, (cent[0] - 5, cent[1] - 5), 255)
    # Repeat with slightly different seed point if image is unchanged
    if cv2.countNonZero(cv2.subtract(mask_v, mask_v_orig)) == 0:  
        print('yes')
        cv2.floodFill(mask_v, None, (cent[0] - 7, cent[1] - 3), 255)
        if cv2.countNonZero(cv2.subtract(mask_v, mask_v_orig)) == 0:  
            print('yes')
            cv2.floodFill(mask_v, None, (cent[0] - 3, cent[1] - 7), 255)
    
    mask_tl = cv2.bitwise_and(mask, mask_v)
    mask_bl = cv2.bitwise_and(mask, 255-mask_v)
    mask_tr = cv2.bitwise_and(255-mask, mask_v)
    mask_br = cv2.bitwise_and(255-mask, 255-mask_v)

    # mask_tl3 = np.repeat(np.expand_dims(mask_tl, 2), 3, 2)
    # mask_bl3 = np.repeat(np.expand_dims(mask_bl, 2), 3, 2)
    # mask_tr3 = np.repeat(np.expand_dims(mask_tr, 2), 3, 2)
    # mask_br3 = np.repeat(np.expand_dims(mask_br, 2), 3, 2)

    # Split  image into quadrants
    # img_tl = ~mask_tl3 + cv2.bitwise_and(img_c, img_c, mask=mask_tl)
    # img_bl = ~mask_bl3 + cv2.bitwise_and(img_c, img_c, mask=mask_bl)
    # img_tr = ~mask_tr3 + cv2.bitwise_and(img_c, img_c, mask=mask_tr)
    # img_br = ~mask_br3 + cv2.bitwise_and(img_c, img_c, mask=mask_br)
    img_tl = cv2.bitwise_and(img_c, img_c, mask=mask_tl)
    img_bl = cv2.bitwise_and(img_c, img_c, mask=mask_bl)
    img_tr = cv2.bitwise_and(img_c, img_c, mask=mask_tr)
    img_br = cv2.bitwise_and(img_c, img_c, mask=mask_br)

    ## DEBUG
    if debug:
        using = "MIN AREA BOX" if using_minbox else "BOUNDING BOX"
        print(f"minbox_coverage: {minbox_coverage*100:.2f}%, {pre_alpha:.2f}°, {edge_ratio:.2f} | {using}")
        img = cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        img = cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 2)
        if using_minbox:
            img = cv2.drawContours(img, [rect_pts], -1, (128, 128, 128), 2)
        else:
            img = cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (128, 128, 128), 2)
        img = cv2.circle(img, tuple(cent), 5, (255, 0, 0), 4)
        img = cv2.circle(img, (x0, y0), 5, (255, 0, 0), 4)
        img = cv2.circle(img, (x1, y1), 5, (255, 0, 0), 4)
        img = cv2.circle(img, (x2, y2), 5, (255, 0, 0), 4)
        img = cv2.circle(img, (x3, y3), 5, (255, 0, 0), 4)
    ##
    if save:
        cv2.imwrite(f"viz_chosen_{filename}", img)

    ## EXTRA DEBUG
    # cv2.imshow('mask', mask)
    # cv2.imshow('mask_v', mask_v)
    # cv2.imshow('mask_tl', mask_tl)
    # cv2.imshow('mask_bl', mask_bl)
    # cv2.imshow('mask_tr', mask_tr)
    # cv2.imshow('mask_br', mask_br)
    ## DEBUG
    if debug:
        # cv2.imshow('img_tl', img_tl)
        # cv2.imshow('img_bl', img_bl)
        # cv2.imshow('img_tr', img_tr)
        # cv2.imshow('img_br', img_br)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    ##

    return img_tl, img_tr, img_bl, img_br


if __name__ == "__main__":
    img_urls = [
        'nTEST1.png',
        'nTEST2.png',
        'nTEST3.png',
        'nTEST4.png',
        'nTEST5.png',
        'nTEST6.png',
        'nTEST7.png',
        'nTEST8.png',
        'nTEST9.png',
        'nTEST10.png',
        'nTEST11.png',
        'nTEST12.png',
        'nTEST13.png',
        'nTEST14.png',
        'nTEST15.png',
    ]

    for url in img_urls:
        img_thr = read_image(url)
        divide_sign_mask_quadrants(img_thr, url, debug=True)
    cv2.destroyAllWindows()
