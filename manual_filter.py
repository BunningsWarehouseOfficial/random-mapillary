"""Script for manual filtering of images as follows '[Key]: [Tag]':
'p': PASS
'f': FAIL
'r': RECHECK
'o': OTHER
DEL: Delete
ESC: Quit
All other keys: Skip
"""

import argparse
from glob import glob
import os
import shutil
from tkinter import filedialog
from tkinter import *

import cv2 as cv


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--other-only", action="store_true", help="Only allow the OTHER and Skip actions.")
parser.add_argument("-c", "--copy", action="store_true", help="Copy images instead of moving them.")
args = parser.parse_args()


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv.resize(image, dim, interpolation=inter)

def main():
    root = Tk()
    root.withdraw()
    indir = filedialog.askdirectory(title="Input directory of UNFILTERED images")
    if not args.other_only:
        passdir = filedialog.askdirectory(title="Directory to move images that PASSED the filter")
        faildir = filedialog.askdirectory(title="Directory to move images that FAILED the filter")
        recheckdir = filedialog.askdirectory(title="Directory to move images that need to be RECHECKED")
    otherdir = filedialog.askdirectory(title="Directory to move images that fulfill some OTHER criteria")

    if args.copy:
        fn = shutil.copy
    else:
        fn = shutil.move

    imgs = glob(f"{indir}/*.jpg")
    for ii, img_path in enumerate(imgs):
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        cv.namedWindow(img_path, cv.WINDOW_NORMAL)
        cv.setWindowProperty(img_path, cv.WND_PROP_ASPECT_RATIO, cv.WINDOW_KEEPRATIO)
        cv.setWindowProperty(img_path, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        # img = ResizeWithAspectRatio(img, height=1280)
        cv.imshow(img_path, img)
        cv.moveWindow(img_path, 0, 0)
        pressedKey = cv.waitKey(0) & 0xFF
        if (pressedKey == ord('p') or pressedKey == ord('P')) and not args.other_only:
            fn(img_path, os.path.join(passdir, os.path.basename(img_path)))
            print(f"[{ii + 1}] Marked as PASS: {os.path.basename(img_path)}")
        elif (pressedKey == ord('f') or pressedKey == ord('F')) and not args.other_only:
            fn(img_path, os.path.join(faildir, os.path.basename(img_path)))
            print(f"[{ii + 1}] Marked as FAIL: {os.path.basename(img_path)}")
        elif (pressedKey == ord('r') or pressedKey == ord('R')) and not args.other_only:
            fn(img_path, os.path.join(recheckdir, os.path.basename(img_path)))
            print(f"[{ii + 1}] Marked as RECHECK: {os.path.basename(img_path)}")
        elif pressedKey == ord('o') or pressedKey == ord('O'):
            fn(img_path, os.path.join(otherdir, os.path.basename(img_path)))
            print(f"[{ii + 1}] Marked as OTHER: {os.path.basename(img_path)}")
        # Helpful here: https://www.asciitable.com
        elif pressedKey == 8 or pressedKey == 127:  # Backspace or Delete keys
            os.remove(img_path)
            print(f"[{ii + 1}] Deleted: {os.path.basename(img_path)}")
        elif pressedKey == 27:  # Escape key
            cv.destroyAllWindows()
            break
        else:
            print(f"[{ii + 1}] Skipped: {os.path.basename(img_path)}")
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
