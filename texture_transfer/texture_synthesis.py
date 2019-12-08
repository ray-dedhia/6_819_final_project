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

def im2col_sliding_strided(image, window_size, stepsize=1):
    """
    Generates a sliding window. 
    
    @param:
        window_size (1d array): the size of the sliding window (width, height)
        image (2d array): the image the sliding window goes over
        
    @return:
        out_view: a 2D array where each row of out_view is the sliding window flattened
    """
    # Parameters
    m,n = image.shape
    s0, s1 = image.strides
    nrows = m-window_size[0]+1
    ncols = n-window_size[1]+1
    shp = window_size[0],window_size[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(image, shape=shp, strides=strd)
    return (out_view.reshape(window_size[0]*window_size[1],-1)[:,::stepsize]).T

def find_matches(template, texture, gaussian):
    """
    Calculate the SSD (sum of squared differences) between the non-NAN (filled in) values
    in the template and the corresponding values in every w x w window in the texture, weighted
    by the gaussian of the (w x w) impulse matrix. Return the pixel values at the centers of 
    the w x w windows in the texture with some error (SSD) below some threshhold epsilon, and
    their corresponding errors.
    
    @param:
        - template (3D array): a (w x w x num_channels) block of the values in the 
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
    # error threshhold
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

def synthesize_texture_pixel_by_pixel(texture, w, size):
    """
    Fill the image with the given texture using the algorithm by Efros 
    and Leung, i.e. pixel by pixel.  
    
    @param:
        texture (3D array): the texture; array of shape (height, width, channels)
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
    synth_im[c[0]:c[0]+seed_size,c[1]:c[1]+seed_size,:] = texture[j0:j0+seed_size,i0:i0+seed_size,:]
    
    ### bitmap indicating filled pixels (pixels with non-NAN values)
    filled = np.zeros(size)
    filled[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ] = 1
    n_filled = int(np.sum(filled)) # number of filled pixels
    n_pixels = image_height * image_width # number of total pixels
    
    ### Main Loop
    # next_p init. to 10% of num. of pixels; increments by 10% every time
    # 10% of the pixels are filled
    next_p = n_pixels / 10
    
    # maximum error threshhold
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
        
        # L = [[j0, j1, ..., jn], [i0, i1, ..., in]] = [rows, cols]
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
            
            # only set pixel value if error is less than max threshhold
            if (chosen_error < delta):
                # set the value at (i,j) in image equal to chosen_center_pixel 
                synth_im[jk][ik] = chosen_center_pixel
                # don't need to increase delta
                progress = 1
                # update filled and n_filled
                filled[jk][ik] = 1
                n_filled += 1
                
        # if progress is still 0, increase delta
        if (progress==0):
            delta *= 1.1

    sys.stdout.write("\n") # ends progress bar
        
    return synth_im 

#-------------------------------------------------------------------------------------------

def im2col_sliding_strided(image, window_size, stepsize=1):
    """
    Generates a sliding window. 
    
    @param:
        window_size (1d array): the size of the sliding window (width, height)
        image (2d array): the image the sliding window goes over
        
    @return:
        out_view: a 2D array where each row of out_view is the sliding window flattened
    """
    # Parameters
    m,n = image.shape
    s0, s1 = image.strides
    nrows = m-window_size[0]+1
    ncols = n-window_size[1]+1
    shp = window_size[0],window_size[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(image, shape=shp, strides=strd)
    return (out_view.reshape(window_size[0]*window_size[1],-1)[:,::stepsize]).T

def gen_blocks_inds(output_image, b, overlap):
    """
    Divide the output image into square blocks of size (b x b).
    Output a list containing the blocks' indexes, that is
    [j_first, j_last, i_first, i_last] for each block, such that 
    output_image[j_first:j_last, i_first:i_last, :] is the block.
    """

    blocks = []

    height, width, channels = output_image.shape

    for j in range(overlap, height+overlap, b):
        for i in range(overlap, width+overlap, b):
            delta_x = min((width-i), b) # width-i if block goes off edge, else b
            delta_y = min((height-j), b) # height-j if block goes off edge, else b

            j_first = j
            j_last = j + delta_y
            i_first = i
            i_last = i + delta_x

            blocks.append(np.array([j_first, j_last, i_first, i_last]))
            
    return np.array(blocks)

def add_patch_to_output_image(output_image, block, texture_patch, b, overlap):
    """
    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        texture_patch (3D array): sliding window from the texture; shape is
            (patch_size, patch_size, num_channels)
        b (int): blocks are shape (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        output_image with texture_patch applied to the center of the block,
        with out of bounds pixels ignored
    """

    # get block indices
    j_first, j_last, i_first, i_last = block
    H = j_last - j_first
    W = i_last - i_first

    # last_x and last_y equal overlap plus height and width of block
    last_y = overlap + H 
    last_x = overlap + W
    output_image[j_first:j_last, i_first:i_last, :] = \
        texture_patch[overlap:last_y, overlap:last_x, :]
        
    return output_image

def get_best_patch(output_image, block, all_patches, overlap, patch_size, num_channels):
    """
    Using all_patches (all possible sliding windows over the texture), calculate the
    error between the overlap between each patch and the filled in values of output_image 
    if that patch is chosen.

    @param:
        output_image (3D array): image being synthesized with texture
        block_inds (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        all_patches (4D array): all possible flattened sliding windows over texture of 
            size patch_size x patch_size
        overlap (int): the size of the overlap between the patches when calculating error
        patch_size (int): the size of the patch
        num_channels (int): the number of color channels

    @return:
        the best patch (unflattened)
    """

    # get block indexes
    j_first, j_last, i_first, i_last = block

    # get patch from output image
    output_image_patch = np.array(output_image[j_first-overlap:j_last, 
            i_first-overlap:i_last, :])

    # resize patches if too small
    if (output_image_patch.shape != (patch_size, patch_size, num_channels)):
        output_image_patch.resize((patch_size, patch_size, num_channels))

    # flatten and stack for calculating error with texture patches
    # output image
    flat_output_image_patch = np.reshape(output_image_patch, (patch_size*patch_size, num_channels))
    stacked_flat_output_image_patch = [flat_output_image_patch for i in range(len(all_patches))]

    # overlap error between output image and all texture patches over block
    # calculate sum of squared errors and sum over color channels
    # each index corresponds to the error of a texture patch
    stacked_errors = np.nansum(np.nansum(np.square(np.subtract( \
        stacked_flat_output_image_patch, all_patches)), axis=1), axis=1)

    # get min non-zero error
    min_error = np.min(stacked_errors[stacked_errors>0])

    # make list of patches with error below some threshhold
    epsilon = 0.1 # error threshhold
    valid_texture_patches = []
    for k in range(len(stacked_errors)):
        if ((stacked_errors[k] > 0 and stacked_errors[k] < min_error * (1 + epsilon))):
            valid_texture_patches.append(all_patches[k])
    
    # return random patch from valid patches
    return np.reshape(valid_texture_patches[np.random.randint(len(valid_texture_patches))],
        (patch_size, patch_size, num_channels))

def synthesize_texture_in_patches(texture, b, overlap, size):
    """
    Fill the image with the given texture in pathes.
    
    Using algorithm by Efros and Freeman.

    1. Divide the output image into square blocks of size (b x b)
    1. Row 0, Col 0: Pick a ((b + 2*overlap) x (b + 2*overlap)) random patch from the texture 
       source image and add the center (b x b) pixels to the top-left (b x b) block 
       in the synthesized image (ignore the pixels that go over the edge of the output image)
    2. For each block in the output image:
            a) Iterate through all possible ((b + overlap) x (b + overlap)) patches 
               in the texture source image 
            b) Calculate the error between the overlapping areas and the filled in 
               areas of the output image
            c) Add the patch with the minimum overlap error to the center of the block  
               (ignoring pixels that go over the edge of the output image)
    
    @param:
        texture (filename): the texture image filename
        b (int): the shape of the blocks in the output image is (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error
        size (tuple): height and width of output image

    @return:
        synthesized image as numpy array
    """
    ## Normalize the pixel intensity of the texture image
    texture = im2double(texture)
    [texture_height, texture_width, texture_num_channels] = texture.shape

    ## Initialize the output image (padded with overlap on all sides) to NANs
    output_height, output_width = size
    output_image = np.full((output_height + 2*overlap, output_width + 2*overlap, 
        texture_num_channels), np.nan)

    ## Generate block indexes
    all_blocks_inds = gen_blocks_inds(output_image, b, overlap)

    ## Generate flattened sliding window over texture
    # shape: (num_windows, patch_size*patch_size, num_channels)
    patch_size = b + overlap # same size as block plus overlap on top and left sides
    all_patches = np.dstack([im2col_sliding_strided(texture[:,:,i], 
        (patch_size,patch_size)) for i in range(texture_num_channels)])

    ## Initialization: pick a random (patch_size x patch_size) patch from the texture
    ## source image and place it in the top-left (b x b) block in the output image
    rand_flat_patch = all_patches[np.random.randint(len(all_patches))]
    # reshape patch to make 3D
    random_texture_patch = np.dstack([np.reshape(rand_flat_patch[:,i], 
        (patch_size, patch_size)) for i in range(texture_num_channels)])
    output_image = add_patch_to_output_image(output_image, all_blocks_inds[0], 
        random_texture_patch, b, overlap)

    start_time = time.clock()

    ## Fill in the rest of the blocks
    for i in range(1, len(all_blocks_inds)):
        print_progress(start_time, (i-1) / (len(all_blocks_inds)-1))
        best_patch = get_best_patch(output_image, all_blocks_inds[i], all_patches,
            overlap, patch_size, texture_num_channels)
        output_image = add_patch_to_output_image(output_image, all_blocks_inds[i], 
            best_patch, b, overlap)

    # return output_image with overlap padding cropped out
    return output_image[overlap:-overlap, overlap:-overlap, :]


def run(texture_filename, output_filename, pixel_by_pixel):
    """
    Inputs:
        texture_name (string): name of texture; one of the following:
            ["texture", "rings"]
        output_filename (string): name to save file to; do not include file extension; adds .png
        pixel_by_pixel (boolean): whether to use pixel_by_pixel (True) method or (False) patches method

    Saves synthesized texture to <output_filename>.png
    """

    texture_image = cv2.imread(texture_filename)
    output_image_size = [100, 100]

    if pixel_by_pixel:
        window_size = 5
        output = synthesize_texture_pixel_by_pixel(texture_image, window_size, output_image_size)
    else:
        block_size = 10
        overlap = 5
        output = synthesize_texture_in_patches(texture_image, block_size, overlap, output_image_size)

    plt.figure() 
    plt.imshow(output) 
    plt.axis('off') 
    plt.savefig(output_filename)

run("styles/brickwall.png", "synthesis_output/block_brickwall_2.png", False)
