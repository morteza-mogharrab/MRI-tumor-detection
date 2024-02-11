import numpy as np
import cv2
from skimage.measure import regionprops
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def sym_demo():
    """
    This function processes an input image by calling three 
    key functions: 'identify_skull,' 'score,' and 'find_largest_decreasing_segment.' 
    These functions perform tasks like skull detection, segmentation, 
    Bhattacharyya coefficient calculation for region similarity assessment, 
    and locating potential tumors. Detected parts are then visualized.
    """
    # Request input image from user 
    root = tk.Tk()
    root.withdraw()

    # Read image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp")])
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Normalizing the input image between 0 and 255
    image = (image - np.min(image)) * (255 / (np.max(image) - np.min(image)))

    # Calling identify_skull function that is responsible for preprocessing parts.
    M, I = identify_skull(image)

    # Plotting the vertical (rotated) gray-scale image and its segmented mask
    plt.figure(1)
    # Plot MR Image
    plt.subplot(1, 2, 1)
    plt.imshow(I)
    plt.axis('image')
    # Display the detected skull image
    plt.subplot(1, 2, 2)
    plt.imshow(M, cmap=cm.gray) # Setting ColorMap to grayscale
    plt.axis('image')

    """
    These three lines are important for subsequent calculations 
    and operations as they allow the program to work with 
    the detected skull region.
    """
    STATS = regionprops(M.astype(int))
    midx = round(STATS[0].centroid[1])
    M = M.astype(bool)

    # Displaying the original image and the segmentation 
    plt.figure(2, figsize=(11, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.axis('image')
    plt.title('MRI Skull Image')
    plt.draw()

    plt.subplot(2, 2, 2)
    plt.imshow(I, cmap='gray')
    plt.axis('image')
    plt.title('Segmented Skull & Middle Line')
    plt.axvline(x=midx, ymin=0, ymax=1, color='r', linewidth=2)
    
    """
    This function (cv2.Canny) is crucial for generating the cyan border around 
    the skull, enabling the visualization of the entire skull area 
    highlighted in cyan color. Without this function, the cyan border 
    would be absent, hindering the complete visualization of the skull region.
    """
    edges = cv2.Canny(M.astype(np.uint8), threshold1=0, threshold2=1, apertureSize=3)
    b_x, b_y = np.nonzero(edges)
    plt.plot(b_y, b_x, 'c.', linewidth=1)

    # Creating images and masks
    Im = I[:, midx - 1::-1] # left image, we call this original/test image
    ImMask = M[:, midx - 1::-1] # mask for original image
    RefI = I[:, midx:] # reference image, here it is the right side
    RefIMask = M[:, midx:] # mask for reference image

    # Start of the vertical scan and end of the vertical scan.
    bbox = STATS[0]['BoundingBox']
    starti = round(bbox[1])
    endi = round(bbox[1] + bbox[3])

    # Top-down search: Computing the Bhattacharya coefficient-based score function
    fact = 16  # Histogram binsize
    offset = 25
    BC_diff_TD = score(Im, RefI, ImMask, RefIMask, starti, endi, fact, offset)

    # Score plot of vertical direction
    plt.subplot(2, 2, 3)
    plt.title('Vertical Analysis Plot')
    plt.plot(range(starti, endi + 1), BC_diff_TD)
    vert_scale = 30  # Scale for finding maxima and minima of the vertical score function.
    topy1, downy1 = find_largest_decreasing_segment(BC_diff_TD, vert_scale)
    topy = topy1[0]
    downy = downy1[0]

    plt.subplot(2, 2, 3)
    plt.plot(topy + starti, BC_diff_TD[topy], 'r.', markersize=10)
    plt.plot(downy + starti, BC_diff_TD[downy], 'm.', markersize=10)
    topy = topy + starti
    downy = downy + starti

    # Left-Right search: images and their masks
    # Take transpose of images and masks
    Im = Im[topy:downy, :].T
    ImMask = ImMask[topy:downy, :].T
    RefI = RefI[topy:downy, :].T
    RefIMask = RefIMask[topy:downy, :].T

    # Start of the horizontal scan and end of the horizontal scan.
    startj = 1
    endj = int(min(bbox[0] + bbox[2] - midx + 1, midx - bbox[0] + 1))
    # Computing the Bhattacharya coefficient-based score function
    BC_diff_LR = score(Im, RefI, ImMask, RefIMask, startj, endj, fact, offset)
    horz_scale = 30  # Scale for finding maxima and minima of the horizontal score function.
    leftx1, rightx1 = find_largest_decreasing_segment(BC_diff_LR, horz_scale)

    leftx = leftx1[0]
    rightx = rightx1[0]
    leftx2 = leftx1[0]
    rightx2 = rightx1[0]
    leftx += midx + startj
    rightx += midx + startj
    m_right = np.mean(I[topy:downy, leftx:rightx])  # Right side of the line of symmetry
    m_left = np.mean(I[topy:downy, 2 * midx - rightx:2 * midx - leftx])
    isleft = 0
    if m_left > m_right:
        leftx1 = 2 * midx - rightx
        rightx1 = 2 * midx - leftx
        leftx = leftx1
        rightx = rightx1
        isleft = 1

    # Rearrange the length of BC_diff
    range_length = len(range(midx + startj, midx + endj))
    BC_diff_LR = BC_diff_LR[:range_length]

    # Display the results
    plt.figure(2)
    plt.subplot(2, 2, 4)
    plt.title('Horizontal Analysis Plot')
    if isleft == 1:
        plt.plot(range(midx - endj, midx - startj), -BC_diff_LR[::-1], 'r')
        plt.plot(rightx, -BC_diff_LR[leftx2], 'y.', leftx, -BC_diff_LR[rightx2], 'c.')
    else:
        plt.plot(range(midx + startj, midx + endj), BC_diff_LR, 'r')
        plt.plot(leftx, BC_diff_LR[leftx2], 'c.', rightx, BC_diff_LR[rightx2], 'y.')

    # This part will draw the detected tumor region on the first plotted image (2,2,1).
    plt.subplot(2, 2, 1)
    plt.plot([leftx, rightx], [topy, topy], 'r')
    plt.plot([leftx, rightx], [downy, downy], 'g')
    plt.plot([leftx, leftx], [topy, downy], 'c')
    plt.plot([rightx, rightx], [topy, downy], 'y')
    plt.show()


def identify_skull(I):
    """
    This function identifies the skull in the input image 'I' by preprocessing it.
    """
    # Setting the intensity values of pixels along the image borders to 0.
    # One important tip is that Matlab index starts from 1 but in Python it starts from 0
    I[:, 0] = 0
    I[:, -1] = 0
    I[0, :] = 0
    I[-1, :] = 0

    # Applying segmentation using thresholding method
    th, im_th = cv2.threshold(I.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)

    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    K = im_th | im_floodfill_inv

    """
    Identifying the skull's center and its vertical axis, 
    determining the required degree of rotation. Subsequently, 
    this information is utilized to rotate the skull image, aligning 
    it to the vertical orientation.
    """
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(K, connectivity=4)
    max_area = 0
    max_index = 0
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area > max_area:
            max_area = area
            max_index = i
    L = (labels == max_index)
    ellipse = cv2.fitEllipse(np.argwhere(L == 1))
    orientation = ellipse[2]

    # Extracting the skull region and rotating the image to align with the vertical orientation
    _, _, _, centroids = cv2.connectedComponentsWithStats(L.astype(np.uint8), connectivity=4)
    x0 = int(round(centroids[1][0]))
    y0 = int(round(centroids[1][1]))

    h, w = I.shape
    min_y = min(y0, h - y0) - 1
    min_x = min(x0, w - x0) - 1
    I = I[y0 - min_y + 1: y0 + min_y + 1, x0 - min_x + 1: x0 + min_x + 1]
    L = L[y0 - min_y + 1: y0 + min_y + 1, x0 - min_x: x0 + min_x + 1]

    # Calculatation of rotation angle based on orientation
    if orientation < 0:
        angle = -90 - orientation
    else:
        angle = 90 - orientation

    # Rotating the skull region to align it with the vertical orientation
    M = cv2.warpAffine(L.astype(np.uint8), cv2.getRotationMatrix2D((x0, y0), angle, 1.0), (L.shape[1], L.shape[0]))
    I = cv2.warpAffine(I.astype(np.uint8), cv2.getRotationMatrix2D((x0, y0), angle, 1.0), (I.shape[1], I.shape[0]))

    # Resizing the binary mask and the rotated skull region to match the original image size
    L = cv2.resize(L.astype(np.uint8), (I.shape[1], I.shape[0]))
    M = cv2.resize(M.astype(np.uint8), (I.shape[1], I.shape[0]))
    
    # Returning the binary mask and the rotated image with the detected skull
    return M, I


def score(Im, RefI, ImMask, RefIMask, starti, endi, fact, offset):
    """
    This Function is responsible for calculating Bhattacharya coefficient.
    """
    # Initialize an array to store the Bhattacharya coefficient-based score function
    BC_diff_TD = np.zeros(endi - starti + 1)
    # Determining the minimum and maximum values for histogram bin calculation
    minval = max(np.min(Im), np.min(RefI))
    maxval = min(np.max(Im), np.max(RefI))
    xbins = np.arange(minval, maxval + fact, fact)

    # Iterating through the vertical scan range
    for i in range(starti, endi + 1):
        Tmp = Im[0:i, :]
        H_leftTop, _ = np.histogram(Tmp[ImMask[0:i, :]], bins=xbins)
        Tmp = RefI[0:i, :]
        H_rightTop, _ = np.histogram(Tmp[RefIMask[0:i, :]], bins=xbins)
        Tmp = Im[i:, :]
        H_leftBottom, _ = np.histogram(Tmp[ImMask[i:, :]], bins=xbins)
        Tmp = RefI[i:, :]
        H_rightBottom, _ = np.histogram(Tmp[RefIMask[i:, :]], bins=xbins)

        # Normalize the histograms
        H_leftTop = H_leftTop / np.sum(H_leftTop) + np.finfo(float).eps
        H_rightTop = H_rightTop / np.sum(H_rightTop) + np.finfo(float).eps
        H_leftBottom = H_leftBottom / np.sum(H_leftBottom) + np.finfo(float).eps
        H_rightBottom = H_rightBottom / np.sum(H_rightBottom) + np.finfo(float).eps

        # Computing BCs for different regions
        BC_Top = np.sum(np.sqrt(H_leftTop * H_rightTop))
        BC_Bottom = np.sum(np.sqrt(H_leftBottom * H_rightBottom))

        # Compute difference of BCs
        if i <= starti + offset:
            BC_diff_TD[i - starti] = -BC_Bottom
            if i == starti + offset:
                BC_diff_TD[: i - starti + 1] += BC_Top
        elif i >= endi - offset:
            if i == endi - offset:
                to_subs = BC_Bottom
            BC_diff_TD[i - starti] = BC_Top - to_subs
        else:
            BC_diff_TD[i - starti] = BC_Top - BC_Bottom
    # Returning the Bhattacharya coefficient-based score function
    return BC_diff_TD


def find_largest_decreasing_segment(score, scale):
    """
    This function finds the largest decreasing segment 
    from the score function, meaning that the area with the 
    largest decreasing value is the potential tumor location 
    coordinates.
    """
    # Determining the regional minima and maxima
    hf_scale = int(round(scale / 2))
    # Extending the score function by padding it with half scale elements on both sides
    ext_score = np.concatenate(([score[0]] * hf_scale, score, [score[-1]] * hf_scale))
    N = len(score)
    reg_minmax = np.zeros((N, 1))

    # Iterating through the score function to find regional minima and maxima
    for n in range(N):
        if np.min(ext_score[n:n + 2 * hf_scale + 1]) == score[n]:
            reg_minmax[n] = -1
        elif np.max(ext_score[n:n + 2 * hf_scale + 1]) == score[n]:
            reg_minmax[n] = 1

    # Finding the largest decreasing segment based on the area (not length)
    n = 0
    thisarea = []
    from_part = []
    to_part = []

    # Iterating through the regional minima and maxima to find largest decreasing segments
    while n < N - 2:
        while reg_minmax[n] < 1 and n < N - 2:
            n += 1
        m = n
        n = n + 1
        while reg_minmax[n] == 0 and n < N - 1:
            n += 1
        if reg_minmax[n] == -1:
            # Calculation area of the current segment and store indices
            thisarea.append(0.5 * (score[m] - score[n]) * (n - m))
            from_part.append(m)
            to_part.append(n)

    # Sort the segments based on area in descending order
    ind = np.argsort(thisarea)[::-1]
    from_part = np.array(from_part)[ind]
    to_part = np.array(to_part)[ind]

    # Returning the indices representing the start and end of the largest decreasing segments
    return from_part, to_part


sym_demo()
