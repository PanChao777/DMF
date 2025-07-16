"""
# > Implementation of the classic paper by Zhou Wang et. al.: 
#     - Image quality assessment: from error visibility to structural similarity
#     - https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1284395
# > Maintainer: https://github.com/xahidbuffon

"""
from __future__ import division
import numpy as np
import math
from scipy.ndimage import gaussian_filter



import numpy as np
from scipy.ndimage import gaussian_filter
import math

def getSSIM(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert X.shape == Y.shape, "Image patches provided have different dimensions"
    nch = 1 if X.ndim == 2 else X.shape[-1]
    mssim = []
    for ch in range(nch):
        Xc, Yc = X[..., ch].astype(np.float64), Y[..., ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)

def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # Variables are initialized as suggested in the paper
    K1 = 0.001
    K2 = 0.08
    sigma = 1.5
    win_size = 11  # Changed from 5 to 11 as per Wang et al.'s recommendation
    
    # Means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # Variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # Normalize by unbiased estimate of std dev 
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # Eq. 4 of the paper
    vx = (uxx - ux * ux) * unbiased_norm
    vy = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # Compute SSIM (Eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim / D
    mssim = SSIM.mean()

    return mssim

def getPSNR(X, Y):
    """
       Computes the Peak Signal-to-Noise Ratio between two images.
    """
    # Ensure data is in the range [0, 255]
    target_data = np.array(X, dtype=np.float64)
    ref_data = np.array(Y, dtype=np.float64)
    
    # If images are normalized, convert back to [0, 255]
    if target_data.max() <= 1.0:
        target_data *= 255.0
    if ref_data.max() <= 1.0:
        ref_data *= 255.0
        
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0: 
        return 100
    else: 
        return 20 * math.log10(255.0 / rmse)

