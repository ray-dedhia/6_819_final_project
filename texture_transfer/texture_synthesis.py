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
        out_view: a 2D array where each row of out_view.transpose is the sliding window flattened
    """
    # Parameters
    m,n = image.shape
    s0, s1 = image.strides
    nrows = m-window_size[0]+1
    ncols = n-window_size[1]+1
    shp = window_size[0],window_size[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(image, shape=shp, strides=strd)
    return out_view.reshape(window_size[0]*window_size[1],-1)[:,::stepsize]

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

def gen_patches(output_image, texture, patch_size):
    """
    Gen and return all possible sliding windows over texture of size patch_size x patch_size
    over texture source image. Return list of all patches.

    @param:
        output_image (3D array): image being synthesized with texture
        texture (3D array): the texture
        patch_size (int): the size of the patches is (patch_size, patch_size, channels)
    
    @return:
        all_patches (4D array): all possible sliding windows over texture of 
            size patch_size x patch_size over texture source image. shape is
            (num_patches, patch_size, patch_size, channels)
    """

    patches = []

    height, width, channels = texture.shape
    
    for j in range(0, height-patch_size):
        for i in range(0, width-patch_size):
            patches.append(texture[j:j+patch_size, i:i+patch_size, :])
            #print("patch shape", np.array(patches[-1]).shape)

    return patches

def gen_blocks(output_image, b):
    """
    Divide the output image into square blocks of size (b x b).
    Output a list containing the block=[j_first, j_last, i_first, i_last]
    of each block, such that 
    output_image[j_first:j_last, i_first:i_last, :] is the block
    """

    blocks = []

    height, width, channels = output_image.shape

    for j in range(0, height, b):
        for i in range(0, width, b):
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
        texture_patch (3D array): the values being added to the output image; 
            shape=(patch_size, patch_size, channels)
        b (int): blocks are shape (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        output_image with texture_patch applied to the center of the block,
        with out of bounds pixels ignored
    """

    j_first, j_last, i_first, i_last = block
    H = j_last - j_first
    W = i_last - i_first
    #print("H", H)
    #print("W", W)
    #print("b", b)

    # last_x and last_y equal overlap plus height and width of block
    last_y = overlap + H 
    last_x = overlap + W
    #print("overlap", overlap)
    #print("last y", last_y)
    #print("last x", last_x)
    #print("output image block shape", np.array(output_image[j_first:j_last, i_first:i_last, :]).shape)
    #print("texture patch shape", np.array(texture_patch).shape)
    #print("texture patch cropped shape", np.array(texture_patch[overlap:last_y, overlap:last_x, :]).shape)
    output_image[j_first:j_last, i_first:i_last, :] = texture_patch[overlap:last_y, overlap:last_x, :]
        
    return output_image

def get_overlap_error(output_image, block, patch, overlap):
    """
    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        patch (3D array): one of all possible sliding windows over texture of 
            size patch_size x patch_size over texture source image. shape is
            (patch_size, patch_size, channels)
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        error_sum (int): sum of squared errors
    """

    # get overlap type from block 
    j_first, j_last, i_first, i_last = block
    top_edge_overlap = (j_first != 0)
    left_edge_overlap = (i_first != 0)

    # calculate error
    error_sum = 0

    if (top_edge_overlap):
        x_start = i_first - overlap if left_edge_overlap else i_first
        x_end = i_last
        y_start = j_first - overlap 
        y_end = j_first
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                error_sum += np.nansum(np.square(np.subtract(output_image[j][i], patch[j-y_start][i-x_start])))
    
    if (left_edge_overlap):
        x_start = i_first - overlap
        x_end = i_last
        y_start = j_first - overlap if top_edge_overlap else j_first
        y_end = j_last
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                error_sum += np.nansum(np.square(np.subtract(output_image[j][i], patch[j-y_start][i-x_start])))
        
    return error_sum

def get_best_patch(output_image, block, all_patches, texture, overlap):
    """
    Using all_patches (all possible sliding windows of size patch_size placed over texture), calculate error
    between overlap with output_image_block if patch is placed in output_image by output_image_block

    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        all_patches (4D array): all possible sliding windows over texture of 
            size patch_size x patch_size over texture source image. shape is
            (num_windows, patch_size, patch_size, channels)
        texture (3D array): the texture; array of shape (height, width, channels)
        overlap (int): the size of the overlap between the patches when calculating error
    """

    # get sum of squared errors
    errors = []
    for patch in all_patches:
        error = get_overlap_error(output_image, block, patch, overlap)
        errors.append(error)

    # get min non-zero error
    errors = np.array(errors)
    min_error = np.min(errors[errors>0])

    # make list of patches with error below some threshhold
    epsilon = 0.1 # error threshhold
    valid_texture_patches = []
    for k in range(len(errors)):
        if ((errors[k] > 0 and errors[k] < min_error * (1 + epsilon))):
            valid_texture_patches.append(all_patches[k])
    
    # return random patch from valid patches
    return valid_texture_patches[np.random.randint(len(valid_texture_patches))]

def synthesize_texture_in_patches(texture, b, overlap, size):
    """
    Fill the image with the given texture using the algorithm by Efros and Freeman,
    i.e. in patches.

    1. Divide the output image into square blocks of size (b x b)
    1. Row 0, Col 0: Pick a ((b + 2*overlap) x (b + 2*overlap)) random patch from the texture 
       source image and add the center (b x b) pixels to the top-left (b x b) block 
       in the synthesized image (ignore the pixels that go over the edge of the output image)
    2. For each block in the output image:
            a) Iterate through all possible ((b + 2*overlap) x (b + 2*overlap)) patches 
               in the texture source image 
            b) Calculate the error between the overlapping areas and the filled in 
               areas of the output image
            c) Add the patch with the minimum overlap error to the center of the block  
               (ignoring pixels that go over the edge of the output image)
    
    @param:
        texture (3D array): the texture; array of shape (height, width, channels)
        b (int): the shape of the blocks in the output image is (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error
        size (1D array): [image_height, image_width] of image to be generated

    @return:
        synthesized image as numpy array
    """
    ## Normalize the pixel intensity
    texture = im2double(texture)
    [texture_height, texture_width, num_channels] = texture.shape
    image_height = size[0]
    image_width = size[1]
    
    ## Initialize the output image to NANs
    output_image = np.full((image_height, image_width, num_channels),np.nan)

    ## Generate blocks
    blocks = gen_blocks(output_image, b)

    ## Generate all patch windows
    patch_size = b + overlap # same size as block plus overlap on top and left sides
    all_patches = gen_patches(output_image, texture, patch_size)

    ## Initialization: pick a random (patch_size x patch_size) patch from the texture
    ## source image and place it in the top-left (b x b) block in the output image
    #print("texture_width", texture_width)
    #print("texture_height", texture_height)
    #print("patch_size", patch_size)
    j0 = np.random.randint(texture_height-patch_size)
    i0 = np.random.randint(texture_width-patch_size)
    #print("i0, j0", i0, j0)
    random_texture_patch = texture[j0:j0+patch_size, i0:i0+patch_size, :]
    #print("random texture patch shape", np.array(random_texture_patch).shape)
    output_image = add_patch_to_output_image(output_image, blocks[0], random_texture_patch, b, overlap)

    start_time = time.clock()

    ## Fill in the rest of the blocks
    for i in range(1, len(blocks)):
        print_progress(start_time, (i-1) / (len(blocks)-1))
        best_texture_patch = get_best_patch(output_image, blocks[i], all_patches, texture, overlap)
        output_image = add_patch_to_output_image(output_image, blocks[i], best_texture_patch, b, overlap)

    return output_image

def run(texture_name, output_filename, pixel_by_pixel):
    """
    Inputs:
        texture_name (string): name of texture; one of the following:
            ["texture", "rings"]
        output_filename (string): name to save file to; do not include file extension; adds .png
        pixel_by_pixel (boolean): whether to use pixel_by_pixel (True) method or (False) patches method

    Saves synthesized texture to <output_filename>.png
    """

    filename = "styles/" + texture_name + ".png"
    image = cv2.imread(filename)
    output_image_size = [100, 100]

    if pixel_by_pixel:
        window_size = 5
        target = synthesize_texture_pixel_by_pixel(image, window_size, output_image_size)
    else:
        block_size = 80
        overlap = 5
        target = synthesize_texture_in_patches(image, block_size, overlap, output_image_size)

    plt.figure() 
    plt.imshow(target) 
    plt.axis('off') 
    plt.savefig("output/" + output_filename + ".png")

run("brickwall", "block_brickwall", False)
