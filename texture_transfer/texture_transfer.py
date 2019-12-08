#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import numpy as np
from scipy import ndimage
import math
import cv2
import random
import sys
import math
import matplotlib.pyplot as plt
import time

def rgb2gray(rgb):
    return np.array(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]), dtype="int32")

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
    sys.stdout.write("[{}] {}%, [{}s / {}s]".format(bar_string, 
        int(percent*100), time_passed, est_total_time))
    sys.stdout.flush()

def im2array(filename):
    img = Image.open(filename) 
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

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
    return np.array((out_view.reshape(window_size[0]*window_size[1],-1)[:,::stepsize]).T, dtype="int32")

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

def get_best_patch(output_image, block, all_patches, texture_luminosity_windows,
        target_luminosity, overlap, patch_size, num_channels):
    """
    Using all_patches (all possible sliding windows over the texture), calculate the
    error between the overlap between each patch and the filled in values of output_image 
    if that patch is chosen plus the luminosity error.

    @param:
        output_image (3D array): image being synthesized with texture
        block_inds (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        all_patches (4D array): all possible flattened sliding windows over texture of 
            size patch_size x patch_size
        texture_luminosity_windows (3D array): all possible windows of size (patch_size x patch_size)           over the luminosity (grayscale) map of the texture image 
        target_luminosity (2D array): the luminosity (grayscale) map of the target image 
        overlap (int): the size of the overlap between the patches when calculating error
        patch_size (int): the size of the patch
        num_channels (int): the number of color channels

    @return:
        the best patch (unflattened)
    """

    # alpha - how much to weigh luminosity
    alpha = 5

    # get block indexes
    j_first, j_last, i_first, i_last = block

    # get patch from output image
    output_image_patch = np.array(output_image[j_first-overlap:j_last, 
            i_first-overlap:i_last, :], dtype="int32")

    # get patch from target luminosity map
    target_lum_patch = np.array(target_luminosity[j_first-overlap:j_last, 
            i_first-overlap:i_last], dtype="int32")

    # resize patches if too small
    if (output_image_patch.shape != (patch_size, patch_size, num_channels)):
        output_image_patch.resize((patch_size, patch_size, num_channels))
    if (target_lum_patch.shape != (patch_size, patch_size)):
        target_lum_patch.resize((patch_size, patch_size))

    # flatten and stack for calculating error with texture patches
    # output image
    flat_output_image_patch = np.reshape(output_image_patch, (patch_size*patch_size, num_channels))
    stacked_flat_output_image_patch = [flat_output_image_patch for i in range(len(all_patches))]

    # luminosity map
    flat_target_lum_patch = np.reshape(target_lum_patch, (patch_size*patch_size))
    stacked_flat_target_lum_patch = [flat_target_lum_patch for i in range(len(all_patches))]

    # overlap error between output image and all texture patches over block
    # calculate sum of squared errors and sum over color channels
    # each index corresponds to the error of a texture patch
    stacked_overlap_errors = np.array(np.nansum(np.nansum(np.square(np.subtract( \
        stacked_flat_output_image_patch, all_patches)), axis=1), axis=1), dtype="int32")

    # luminosity error between target patch being filled and all texture patches
    stacked_lum_errors = np.array(np.nansum(np.square(np.subtract( \
        stacked_flat_target_lum_patch, texture_luminosity_windows)), axis=1), dtype="int32")

    # sum errors
    stacked_errors = stacked_overlap_errors + stacked_lum_errors

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

def transfer_texture_in_patches(texture, target, b, overlap):
    """
    Fill the image with the given texture in pathes, minimizing the error between the
    luminosity of the texture and target images
    
    Using algorithm by Efros and Freeman.

    1. Divide the output image into square blocks of size (b x b)
    1. Row 0, Col 0: Pick a ((b + 2*overlap) x (b + 2*overlap)) random patch from the texture 
       source image and add the center (b x b) pixels to the top-left (b x b) block 
       in the synthesized image (ignore the pixels that go over the edge of the output image)
    2. For each block in the output image:
            a) Iterate through all possible ((b + overlap) x (b + overlap)) patches 
               in the texture source image 
            b) Calculate the error between the overlapping areas and the filled in 
               areas of the output image plus the error between the luminosity maps
            c) Add the patch with the minimum overlap error to the center of the block  
               (ignoring pixels that go over the edge of the output image)
    
    @param:
        texture (image): the texture image
        target (image): the target image
        b (int): the shape of the blocks in the output image is (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        synthesized image as numpy array
    """
    [texture_height, texture_width, texture_num_channels] = texture.shape
    [target_height, target_width, target_num_channels] = target.shape

    ## Get sliding window over luminosity maps of texture and target images
    patch_size = b + overlap # same size as block plus overlap on top and left sides
    texture_luminosity_windows = im2col_sliding_strided(rgb2gray(texture), (patch_size, patch_size))
    target_luminosity = rgb2gray(target)
    
    ## Initialize the output image (padded with overlap on all sides) to NANs
    output_image = np.full((target_height + 2*overlap, target_width + 2*overlap, 
        target_num_channels), np.nan)

    ## Generate block indexes
    all_blocks_inds = gen_blocks_inds(output_image, b, overlap)

    ## Generate flattened sliding window over texture
    # shape: (num_windows, patch_size*patch_size, num_channels)
    all_patches = np.dstack([im2col_sliding_strided(texture[:,:,i], 
        (patch_size,patch_size)) for i in range(texture_num_channels)])

    ## Initialization: pick a random (patch_size x patch_size) patch from the texture
    ## source image and place it in the top-left (b x b) block in the output image
    j0 = np.random.randint(texture_height-patch_size)
    i0 = np.random.randint(texture_width-patch_size)
    # get random (flattened) patch
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
            texture_luminosity_windows, target_luminosity,
            overlap, patch_size, target_num_channels)
        output_image = add_patch_to_output_image(output_image, all_blocks_inds[i], 
            best_patch, b, overlap)

    # return output_image with overlap padding cropped out
    return output_image[overlap:-overlap, overlap:-overlap, :]

def run(texture_filename, target_filename, output_filename):
    """
    Inputs:
        texture_name (string): name of texture; one of the following:
            ["texture", "rings"]
        output_filename (string): name to save file to; do not include file extension; adds .png

    Saves synthesized texture to <output_filename>.png
    """

    texture_image = im2array(texture_filename)
    target_image = im2array(target_filename)

    block_size = 20
    overlap = 5
    target = transfer_texture_in_patches(texture_image, target_image, block_size, overlap)

    plt.figure() 
    plt.imshow(target) 
    plt.axis('off') 
    plt.savefig(output_filename)

run("styles/starry_night.png", "targets/bridge.png", "transfer_output/texture_starry_night_target_bridge_bsize_20_overlap_5.png")
