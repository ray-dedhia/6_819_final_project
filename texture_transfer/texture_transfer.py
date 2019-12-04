#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy import ndimage
import math
import cv2
import random
import sys
import math
import matplotlib.pyplot as plt
import time

def print_progress(start_time, percent):
    """
    start_time (float): value returned by time.clock() at beginning of algorithm
    percent (float): decimal percentage
    """
    time_passed = time.clock() - start_time
    if percent==0:
        est_total_time = 0
        time_passed = int(time_passed)
    else:
        est_total_time = int(time_passed / percent)
        time_passed = int(time_passed)

    total_bar_size = 50
    bar_size = int(percent * total_bar_size)
    bar_string = "="*bar_size + (total_bar_size-bar_size)*" "
    sys.stdout.write("\r") # flush progress bar
    sys.stdout.write("[{}] {}%, [{}s / {}s]".format(bar_string, int(percent*100), time_passed, est_total_time))
    sys.stdout.flush()

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

def resize_array(array, new_size):
    """
    adds array to top-left of np.zeros(new_size)
    """
    a_width, a_height = array.shape
    new_width, new_height = new_size
    
    new_array = np.zeros(new_size)
    
    for i in range(a_width):
        for j in range(a_height):
            new_array[i][j] = array[i][j]
            
    return new_array

def im2col_sliding_strided(A, BSZ, stepsize=1):
    """
    Generates a sliding window. 
    
    @param:
        BSZ (1d array): the size of the sliding window (width, height)
        A (2d array): the matrix the sliding window goes over
        
    @return:
        A: 2D array where each row of A.transpose is the sliding window flattened
    """
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides    
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def find_matches(template, texture, gaussian):
    """
    Calculate the SSD (sum of squared differences) between the non-NAN (filled in) values
    in the template and the corresponding values in every w x w window in the texture, weighted
    by the gaussian of the (w x w) impulse matrix. Return the pixel values at the centers of 
    the w x w windows in the texture with some error (SSD) below some threshold epsilon, and
    their corresponding errors.
    
    @param:
        - template (3D array): a (w x w x num_channels) section of the values in the 
            image centered around a pixel in boundary, which is an array of the 
            indices of the pixels to be filled in
        - texture (3D array): the entire texture
        - gaussian (3D array): 3 stacked 2D gaussians; the gaussian with standard deviation 
            (w/6.4) of the (w x w) impulse matrix; used to calculate the SSD; 
            a (w x w x num_channels) array
            
    @return:
        - valid_centers (2D array): (R, G, B) pixel values at centers of (w x w) windows in texture
        - errors (1D array): SSD[k] for all SSD[k] such that SSD[k] < min(SSD) * (1 + epsilon)
    """
    # error threshold
    epsilon = 0.1 
    
    # get w from gaussian
    w = gaussian.shape[0]
    
    # valid_mask is a w x w x num_channels mask with ones 
    # where template has non-NAN values (has been filled)
    valid_mask = np.ones(np.shape(template))
    valid_mask[np.isnan(template)] = 0
    
    # gaussian_mask is gaussian multiplied (element-wise) by valid_mask then normalized
    gaussian_mask = np.multiply(gaussian,valid_mask)
    gaussian_mask = gaussian_mask / np.sum(gaussian_mask)
    
    # get height and width of texture, and num_channels
    texture_height, texture_width, num_channels = texture.shape
    
    # create sliding windows
    # transpose so each row corresponds to a flattened (w,w) array
    # shape: (num_windows, w*w, num_channels)
    windows = np.dstack([im2col_sliding_strided(texture[:,:,i], (w,w)).T for i in range(num_channels)])
    
    # flatten and tile/stack gaussian_mask and template
    # transpose so that each row corresponds to a flattened (w,w) array
    # shape of both: (num_windows, w*w, num_channels)
    flat_gaussian_masks = np.array([np.ndarray.reshape(gaussian_mask, (w**2, num_channels)) 
                                    for i in range(windows.shape[0])])
    flat_templates = np.array([np.ndarray.reshape(template, (w**2, num_channels)) 
                               for i in range (windows.shape[0])])
    
    # calculate SSDs
    # nansum treats nans as zeros
    # shape: (num_windows, num_channels) (one SSD error per window)
    SSDs = np.nansum(np.multiply(np.square(np.subtract(windows, flat_templates)), flat_gaussian_masks), axis=1)
    # sum SSDs over num_channels
    SSDs = np.nansum(SSDs, axis=1)
    
    # min non-zero SSD
    min_SSD = np.min(SSDs[SSDs>0])
        
    # pixel values at centers of (w x w, num_channels) windows in texture with 0 < SSD[k] < epsilon
    # (greater than 0 because ignoring nans)
    valid_center_pixels = []
    valid_SSDs = []
    for k in range(len(SSDs)):
        if ((SSDs[k] > 0 and SSDs[k] < min_SSD * (1 + epsilon))):
            valid_center_pixels.append(windows[k][w**2//2])
            valid_SSDs.append(SSDs[k])
            
    return (valid_center_pixels, valid_SSDs)

def synthesize_texture(texture, w, size):
    """
    Fill the image with the given texture using the algorithm by Efros and Leung.  
    
    @param:
        texture (3D array): one color channel of the entire texture
        w (int): the size of the template
        size (1D array): [image_height, image_width] of image to be generated

    @return:
        synthesized image as numpy array
    """

    start_time = time.clock()
    
    ## Normalize the pixel intensity
    texture = im2double(texture)
    seed_size = 3
    [texture_height, texture_width, num_channels] = texture.shape
    image_height = size[0]
    image_width = size[1]
    
    ## initialize the image to be synthesized to NANs
    synth_im = np.full((image_height, image_width, num_channels),np.nan)
    
    # Impulse is an array that is all zeros except the center value, which is one
    # gaussian is the gaussian with standard deviation (w/6.4) and mean (int(w/2))
    # We are using the gaussian to calculate a weighted SSD (sum of squared differences)
    impulse = np.zeros((w,w))
    impulse[w//2][w//2] = 1 ## (n1 // n2) is integer division
    gaussian = ndimage.gaussian_filter(impulse, (w/6.4))
    # make 3D gaussian
    gaussian = np.dstack([gaussian for i in range(num_channels)])
    
    ### Initialization: pick a random 3x3 patch from the texture 
    ### and place it in the middle of the synthesized image.
    # i0 = round(seed_size + np.random.uniform(0,1) * (texture_height - 2 * seed_size))
    # j0 = round(seed_size + np.random.uniform(0,1) * (texture_width - 2 * seed_size))
    i0 = 31
    j0 = 3
    c = [image_height//2, image_width//2] # center of image
    synth_im[c[0]:c[0]+seed_size,c[1]:c[1]+seed_size,:] = texture[i0:i0+seed_size,j0:j0+seed_size,:]
    
    ### bitmap indicating filled pixels (pixels with non-NAN values)
    filled = np.zeros(size)
    filled[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ] = 1
    n_filled = int(np.sum(filled)) # number of filled pixels
    n_pixels = image_height * image_width # number of total pixels
    
    ### Main Loop
    # next_p init. to 10% of num. of pixels; increments by 10% every time
    # 10% of the pixels are filled
    next_p = n_pixels / 10
    
    # maximum error threshold
    delta = 0.3 
    
    # while there are still unfilled pixels
    while(n_filled < n_pixels):
        # if progress still 0 at end of iteration, increase delta
        progress = 0
        
        print_progress(start_time, n_filled/n_pixels)

        # show synthesized image at every 10%
        if (n_filled > next_p):
            plt.imshow(synth_im)
            
        # filled is a array of all the non-NAN values in the image being generated
        # dilation is that array with the NAN values next to ones set to ones as well
        dilation = ndimage.binary_dilation(filled)
        
        # boundary: each 1 value corresponds to a NAN to be filled in the synthesized image
        # in this iteration (ignore 0 values - have been filled or will be filled later)
        boundary = dilation - filled       
        
        # L = [[i0, i1, ..., in], [j0, j1, ..., jn]] = [rows, cols]
        # where each [ik, jk] is an index be filled in this iteration 
        L = np.array(np.nonzero(boundary))
        # locations = [(i0,j0), (i1, j1), ..., (in, jn)]
        locations = L.T
        
        # For each [ik,jk] in bounded locations, template is a w x w array from the image being 
        # generated, centered at (ik,jk) in the image
        for ((ik,jk)) in locations:
            # pad synthesized image with w//2 at borders
            pad_size = w//2
            pad_tuple = ((pad_size, pad_size), (pad_size, pad_size), (0,0))
            padded_image = np.pad(synth_im, pad_tuple, mode='constant')
            # [ik-w//2+pad_size:ik+w//2+pad_size+1, jk-w//2+pad_size:jk+w//2+pad_size+1]
            # is the same as [ik:ik+2*pad_size+1, jk:jk+2*pad_size+1]
            template = padded_image[ik:ik+2*pad_size+1, jk:jk+2*pad_size+1,:]
            
            # centers of (w x w) windows from texture that match the template below a set error
            valid_center_pixels, errors = find_matches(template, texture, gaussian)
            
            # pick a random center from valid centers
            rand_ind = np.random.randint(len(valid_center_pixels))
            chosen_center_pixel = valid_center_pixels[rand_ind]
            chosen_error = errors[rand_ind]
            
            # only set pixel value if error is less than max threshold
            if (chosen_error < delta):
                # set the value at (i,j) in image equal to chosen_center_pixel 
                synth_im[ik][jk] = chosen_center_pixel
                # don't need to increase delta
                progress = 1
                # update filled and n_filled
                filled[ik][jk] = 1
                n_filled += 1
                
        # if progress is still 0, increase delta
        if (progress==0):
            delta *= 1.1

    sys.stdout.write("\n") # ends progress bar
        
    return synth_im 

def run(texture_name, output_filename):
    """
    Inputs:
        texture_name (string): name of texture; one of the following:
            ["texture", "rings"]
        output_filename (string): name to save file to; do not include file extension; adds .png

    Saves synthesized texture to <output_filename>.png
    """
    names_to_textures = {"texture": cv2.imread('texture.jpg'), "rings": cv2.imread('rings.jpg')}
    window_size = 5
    target = synthesize_texture(names_to_textures[texture_name], window_size, [100, 100])

    plt.figure() 
    plt.imshow(target) 
    plt.axis('off') 
    plt.savefig("output/" + output_filename + ".png")

run("texture", "synth_texture")
