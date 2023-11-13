import numpy as np 
import cv2 


def sobel_xy(img, orient='x', thresh=(20, 100)): 
    """
    Define a function that applies Sobel x or y. 
    The gradient in the x-direction emphasizes edges closer to vertical.
    The gradient in the y-direction emphasizes edges closer to horizontal.
    """

    if orient == 'x':
        abs_sobel = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    else : 
        abs_sobel = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    
    # Rescale back to 8-bit Integer 
    scaled_sobel = np.uint8(abs_sobel* 255 / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel) 
    binary_output[(scaled_sobel >= thresh[0] & scaled_sobel <=thresh[1])] = 255 

    return binary_output


# return the Magnitude of Gradient 
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel) 
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    grad_mag = np.sqrt(sobelx**2 + sobely**2) 

    scale_factor = np.max(grad_mag)/255 
    grad_mag = (grad_mag/scale_factor).astype(np.uint8)

    binary_output = np.zeros_like(grad_mag) 
    binary_output[(grad_mag >= mag_thresh[0]) & (grad_mag <= mag_thresh[1])] = 255 

    return binary_output


# Direction of Gradient
def dir_thresh(img, sobel_kernel=3, thresh=(0.7, 1.3)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_graddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(abs_graddir) 
    binary_output[(abs_graddir >= thresh[0] & (abs_graddir <= thresh[1]))] = 255 

    return binary_output.astype(np.uint8)


def ch_thresh(ch, thresh=(80, 255)):
    binary = np.zeros_like(ch) 
    binary[(ch > thresh[0]) & (ch <= thresh[1])] = 255 
    return binary 



def gradient_combines(img, th_x, th_y, th_mag, th_dir):
    """
    FInd lane with gradient Information of Red Channel
    """

    rows, cols = img.shape[:2] # H W C
    R = img[220: rows -12, 0:cols, 2] # BGR 순

    sobelx = sobel_xy(R, 'x', th_x) 
    sobely = sobel_xy(R, 'y', th_y) 

    mag_img = mag_thresh(R, 3, th_mag) 
    dir_img = dir_thresh(R, 15, th_dir)


    gradient_comb = np.zeros_like(dir_img).astype(np.uint8) 
    gradient_comb[( (sobelx >1) & (mag_img >1) & (dir_img > 1)) | ((sobelx >1) & (sobely > 1))] = 255 

    return gradient_comb



def hls_combine(img, th_h, th_l, th_s):
    # convert to hls color space 

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    rows, cols = img.shape[:2] 
    R = img[220:rows -12, 0:cols, 2] 

    _, R = cv2.threshold(R, 180, 255, cv2.THRESH_BINARY) 

    H = hls[220:rows -12, 0:cols, 0]
    L = hls[220:rows -12, 0:cols, 1]
    S = hls[220:rows -12, 0:cols, 2] 

    h_img = ch_thresh(H, th_h) 
    l_img = ch_thresh(L, th_l) 
    s_img = ch_thresh(S, th_s) 


    hls_comb = np.zeros_like(s_img).astype(np.uint8)
    hls_comb[((h_img > 1) & (l_img ==0)) | ((s_img == 0) & (h_img > 1) & (l_img >1))] = 255 
    
    return hls_comb 


def comb_result(grad, hls):
    result = np.zeros_like(hls).astype(np.uint8)
    result[(grad >1)] = 100 
    result[(hls > 1)] = 255 
    return result