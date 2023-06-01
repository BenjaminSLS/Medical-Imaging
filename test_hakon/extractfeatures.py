import numpy as np
import pandas as pd
from skimage import morphology
from scipy.ndimage import rotate
from skimage.segmentation import slic
from skimage.measure import regionprops


# Suppress scientific notation on the numpy numbers
np.set_printoptions(suppress=True)


def extract_features(image, mask):
    image = image[:, :, :3]
    mask[mask > 0] = 1
    image, mask = resize(image, mask)
    image = filter_image(image, mask)
    area = get_area(mask)
    perimeter = get_perimeter(mask)
    compactness = get_compactness(area, perimeter)
    rotation_asymmetry = get_rotation_asymmetry(mask)
    asymmetry = get_asymmetry(mask)
    hue_sd, sat_sd, val_sd = get_color(image)
    diagnostic = np.nan
    mask_area = mask.shape[0] * mask.shape[1]
    return np.array([diagnostic, area / mask_area, perimeter / mask_area, compactness, rotation_asymmetry, asymmetry, hue_sd, sat_sd, val_sd])


def resize(image, mask):
    """Center the image and mask)"""
    buffer = 5
    row_n, col_n = mask.shape

    left = -1
    right = -1
    top = -1
    bottom = -1

    # Top to bottom
    for td in range(row_n):
        if top == -1:
            if np.any(mask[td, :] == 1):
                top = td

        else:
            bottom = td
            if not np.any(mask[td, :] == 1):
                break

    # Left to right
    for lr in range(col_n):

        if left == -1:
            if np.any(mask[:, lr] == 1):
                left = lr

        else:
            right = lr
            if not np.any(mask[:, lr] == 1):
                break

    top = top - buffer if top - buffer > 0 else 0
    bottom = bottom + buffer if bottom + buffer < row_n else row_n
    left = left - buffer if left - buffer > 0 else 0
    right = right + buffer if right + buffer < col_n else col_n

    image = image[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    return image, mask


def get_area(mask):
    """Get the area of the lesion"""
    return np.sum(mask)


def get_perimeter(mask):
    """Get the perimeter of the lesion"""
    erosion_size = 1
    # Defines a disk brush of size erosion_size
    struct_el = morphology.disk(erosion_size)

    # Erodes the mask with the disk brush
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    # Finds the size of the perimeter by subtracting the eroded mask from the original mask
    perimeter = np.sum(mask - mask_eroded)

    return perimeter


def get_compactness(area, perimeter):
    """Get the compactness of the lesion"""
    compactness = perimeter ** 2 / (4 * np.pi * area)
    return compactness


def get_rotation_asymmetry(mask):
    """Get the rotation asymmetry of the lesion"""
    rotated_mask = rotate(mask, 45)
    rotation_asymmetry = get_asymmetry(rotated_mask)

    return rotation_asymmetry


def get_asymmetry(mask):
    """Get the asymmetry of the lesion"""
    h_flip = np.fliplr(mask)
    diff_h_flip = mask-h_flip

    v_flip = np.flipud(mask)
    diff_v_flip = mask - v_flip

    diff_h_area = np.count_nonzero(diff_h_flip)
    diff_v_area = np.count_nonzero(diff_v_flip)

    if diff_h_area == 0 or diff_v_area == 0:
        return 0

    assy_index = 0.5 * ((diff_h_area / get_area(mask)) +
                        (diff_v_area / get_area(mask)))

    return assy_index


def get_color(image):
    """Get the color features of the lesion"""

    segments_slic = slic(image, n_segments=250,
                         compactness=20, start_label=1, sigma=3)

    regions = regionprops(segments_slic, intensity_image=image)

    mean_intensity = [region.mean_intensity for region in regions]

    color_intensity = [mean for mean in mean_intensity if sum(mean) != 0]

    color_mean_hsv = [rgb_to_hsv(
        col[0], col[1], col[2]) for col in color_intensity]

    color_mean_hue = [hsv[0] for hsv in color_mean_hsv]
    color_mean_sat = [hsv[1] for hsv in color_mean_hsv]
    color_mean_val = [hsv[2] for hsv in color_mean_hsv]

    # check these values
    hue_sd = np.std(np.array(color_mean_hue))
    sat_sd = np.std(np.array(color_mean_sat))
    val_sd = np.std(np.array(color_mean_val))

    return hue_sd, sat_sd, val_sd


def rgb_to_hsv(r, g, b):
    """"
    Changes rgb color to hsv
    """
    max_ = max(r, g, b)
    min_ = min(r, g, b)

    diff = max_ - min_

    if max_ == min_:
        h = 0
    elif max_ == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif max_ == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    elif max_ == b:
        h = (60 * ((r-g)/diff) + 240) % 360
    if max_ == 0:
        s = 0
    else:
        s = (diff/max_)*100
    v = max_*100
    return h, s, v


def filter_image(image, mask):
    """Filter the image using the mask"""

    image[mask == 0] = 0
    return image
