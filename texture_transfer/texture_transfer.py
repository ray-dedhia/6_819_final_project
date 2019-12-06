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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

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

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    # Divide all values by the largest possible value in the datatype
    return im.astype(np.float) / info.max 

def gen_patches(output_image, texture, patch_size):
    """
    Gen and return all possible sliding windows over texture of size patch_size x patch_size
    over texture source image. Return list of all patches.

    @param:
        output_image (3D array): image being synthesized with texture
        texture (3D array): the texture
        patch_size (int): the size of the patches is (patch_size, patch_size, channels)
    
    @return:
        Output a list containing the patch=[j_first, j_last, i_first, i_last] of each
        patch, such that texture[j_first:j_last, i_first:i_last, :] is the patch 
    """

    patches = []

    height, width, channels = texture.shape
    
    for j in range(0, height-patch_size):
        for i in range(0, width-patch_size):
            j_first = j
            j_last = j + patch_size
            i_first = i
            i_last = i + patch_size
            patch = [j_first, j_last, i_first, i_last]
            patches.append(patch)

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

def add_patch_to_output_image(output_image, block, texture, patch, b, overlap):
    """
    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        patch (1D array): a list containing [j_first, j_last, i_first, i_last] 
            such that texture[j_first:j_last, i_first:i_last, :] is the texture patch 
        b (int): blocks are shape (b, b, channels)
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        output_image with texture_patch applied to the center of the block,
        with out of bounds pixels ignored
    """

    # get block indices
    j_first_b, j_last_b, i_first_b, i_last_b = block
    H = j_last_b - j_first_b
    W = i_last_b - i_first_b

    # get texture patch
    j_first_p, j_last_p, i_first_p, i_last_p = patch
    texture_patch = texture[j_first_p:j_last_p, i_first_p:i_last_p, :]

    # last_x and last_y equal overlap plus height and width of block
    last_y = overlap + H 
    last_x = overlap + W
    output_image[j_first_b:j_last_b, i_first_b:i_last_b, :] = texture_patch[overlap:last_y, overlap:last_x, :]
        
    return output_image

def get_overlap_error(output_image, block, texture, patch, overlap):
    """
    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        texture (3D array): the texture
        patch (2D array): patch=[j_first, j_last, i_first, i_last] of patch,
            such that texture[j_first:j_last, i_first:i_last, :] is the texture patch 
        overlap (int): the size of the overlap between the patches when calculating error

    @return:
        error_sum (int): sum of squared errors
    """

    # get texture patch
    j_first_p, j_last_p, i_first_p, i_last_p = patch
    texture_patch = texture[j_first_p:j_last_p, i_first_p:i_last_p, :]

    # get overlap type from block 
    j_first_b, j_last_b, i_first_b, i_last_b = block
    top_edge_overlap = (j_first_b != 0)
    left_edge_overlap = (i_first_b != 0)

    # calculate error
    error_sum = 0

    if (top_edge_overlap):
        x_start = i_first_b - overlap if left_edge_overlap else i_first_b
        x_end = i_last_b
        y_start = j_first_b - overlap 
        y_end = j_first_b
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                error_sum += np.nansum(np.square(np.subtract(output_image[j][i], 
                    texture_patch[j-y_start][i-x_start])))
    
    if (left_edge_overlap):
        x_start = i_first_b - overlap
        x_end = i_last_b
        y_start = j_first_b - overlap if top_edge_overlap else j_first_b
        y_end = j_last_b
        for i in range(x_start, x_end):
            for j in range(y_start, y_end):
                error_sum += np.nansum(np.square(np.subtract(output_image[j][i], 
                    texture_patch[j-y_start][i-x_start])))
        
    return error_sum

def get_luminosity_error(target_luminosity, block, texture_luminosity, patch, overlap):
    """
    @param:
        target_luminosity (2D array): maps (x,y) coordinates in target image to luminosity
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        texture_luminosity (2D array): maps (x,y) coordinates in texture image to luminosity
        patch (2D array): patch=[j_first, j_last, i_first, i_last] of patch,
            such that texture[j_first:j_last, i_first:i_last, :] is the texture patch 
    """

    # get target block
    j_first_b, j_last_b, i_first_b, i_last_b = block
    H = j_last_b - j_first_b
    W = i_last_b - i_first_b
    target_lum_block = target_luminosity[j_first_b:j_last_b, i_first_b:i_last_b]

    # get texture block
    j_first_p, j_last_p, i_first_p, i_last_p = patch
    texture_lum_block = texture_luminosity[j_first_p:j_first_p+H, i_first_p:i_first_p+W]

    # get sum of mean squared error
    return np.nansum(np.square(np.subtract(target_lum_block, texture_lum_block)))

def get_best_patch(output_image, block, all_patches, texture, texture_luminosity,
        target_luminosity, overlap):
    """
    Using all_patches (all possible sliding windows of size patch_size placed over texture), 
    calculate error between overlap with output_image_block if patch is placed in 
    output_image by output_image_block

    @param:
        output_image (3D array): image being synthesized with texture
        block (1D array): equals [j_first, j_last, i_first, i_last], such that
            output_image[j_first:j_last, i_first:i_last, :] is the block in
            the output image to be filled in
        all_patches (2D array): a list containing the 
            patch=[j_first, j_last, i_first, i_last] of each path
            patch, such that texture[j_first:j_last, i_first:i_last, :] is the patch 
        texture (3D array): the texture; array of shape (height, width, channels)
        texture_luminosity (2D array): maps (x,y) coordinates in texture image to luminosity
        target_luminosity (2D array): maps (x,y) coordinates in target image to luminosity
        overlap (int): the size of the overlap between the patches when calculating error
    """

    # alpha - how much to weigh luminosity
    alpha = 2

    # get sum of squared errors
    errors = []
    for patch in all_patches:
        # get error between overlap of texture patch and output image
        overlap_error = get_overlap_error(output_image, block, texture, patch, overlap)
        # get error between luminosity of target image and texture patch
        luminosity_error = get_luminosity_error(target_luminosity, block, 
            texture_luminosity, patch, overlap)
        errors.append(overlap_error + alpha*luminosity_error)

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

def synthesize_texture_in_patches(texture, target, b, overlap):
    """
    Fill the image with the given texture in pathes, minimizing the error between the
    luminosity of the texture and target images
    
    Using algorithm by Efros and Freeman.

    1. Divide the output image into square blocks of size (b x b)
    1. Row 0, Col 0: Pick a ((b + 2*overlap) x (b + 2*overlap)) random patch from the texture 
       source image and add the center (b x b) pixels to the top-left (b x b) block 
       in the synthesized image (ignore the pixels that go over the edge of the output image)
    2. For each block in the output image:
            a) Iterate through all possible ((b + 2*overlap) x (b + 2*overlap)) patches 
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
    ## Normalize the pixel intensity of the texture image
    texture = im2double(texture)
    [texture_height, texture_width, texture_num_channels] = texture.shape

    ## Normalize the pixel intensity of the target image
    target = im2double(target)
    [target_height, target_width, target_num_channels] = target.shape

    ## Get luminosity maps of texture and target images
    texture_luminosity = rgb2gray(texture)
    target_luminosity = rgb2gray(target)
    
    ## Initialize the output image to NANs
    output_image = np.full((target_height, target_width, target_num_channels),np.nan)

    ## Generate blocks
    blocks = gen_blocks(output_image, b)

    ## Generate all patch windows
    patch_size = b + overlap # same size as block plus overlap on top and left sides
    all_patches = gen_patches(output_image, texture, patch_size)

    ## Initialization: pick a random (patch_size x patch_size) patch from the texture
    ## source image and place it in the top-left (b x b) block in the output image
    j0 = np.random.randint(texture_height-patch_size)
    i0 = np.random.randint(texture_width-patch_size)
    random_texture_patch = texture[j0:j0+patch_size, i0:i0+patch_size, :]
    output_image = add_patch_to_output_image(output_image, blocks[0], 
        texture, [j0, j0+patch_size, i0, i0+patch_size], b, overlap)

    start_time = time.clock()

    ## Fill in the rest of the blocks
    for i in range(1, len(blocks)):
        print_progress(start_time, (i-1) / (len(blocks)-1))
        best_patch = get_best_patch(output_image, blocks[i], all_patches, texture, 
            texture_luminosity, target_luminosity, overlap)
        output_image = add_patch_to_output_image(output_image, blocks[i], 
            texture, best_patch, b, overlap)

    return output_image

def run(texture_filename, target_filename, output_filename):
    """
    Inputs:
        texture_name (string): name of texture; one of the following:
            ["texture", "rings"]
        output_filename (string): name to save file to; do not include file extension; adds .png

    Saves synthesized texture to <output_filename>.png
    """

    texture_image = cv2.imread(texture_filename)
    target_image = cv2.imread(target_filename)

    block_size = 20
    overlap = 5
    target = synthesize_texture_in_patches(texture_image, target_image, block_size, overlap)

    plt.figure() 
    plt.imshow(target) 
    plt.axis('off') 
    plt.savefig(output_filename)

run("styles/rings.png", "targets/bridge.png", "transfer_output/block_size_10_overlap_5_target_bridge.png")
