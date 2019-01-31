from numpy import argmax, argmin, empty_like, dot
from sklearn.mixture import GMM
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow 
from random import sample

import warnings
warnings.filterwarnings("ignore")

def sample_RGB_from_picture(rgb_image_arr, total_sampled_pixels = 500):
    total_available_pixels = rgb_image_arr.shape[0]*rgb_image_arr.shape[1]
    random_values = sample(range(0, total_available_pixels), total_sampled_pixels)
    rgb_pixel_values = rgb_image_arr.reshape(total_available_pixels,3)[random_values]
    return rgb_pixel_values
    
def rgb2gray(rgb):
    return dot(rgb[...,:3], [0.299, 0.587, 0.114])


def run_GMM(rgb_pixel_values,n_components=15,covariance_type="full"):
    gmm = GMM(n_components,covariance_type="full").fit(rgb_pixel_values)
    gmm_labels = gmm.predict(rgb_pixel_values)
    
    return gmm, gmm_labels

def predict_pixel_GMM_cluster(pixel,gmm):
    
    pixel = pixel.reshape(1,3)
    return argmax(gmm.predict_proba(pixel))


def predict_pixel_GMM_color_MSE(pixel,gmm):
    pixel_MSE = ((gmm.means_ - pixel) ** 2).mean(axis=1)
    return gmm.means_[argmin(pixel_MSE)],argmin(pixel_MSE)


def predict_pixel_GMM_color_MSE_gray(pixel,gmm):
    pixel_MSE = ((rgb2gray(gmm.means_) - rgb2gray(pixel)) ** 2)
    return gmm.means_[argmin(pixel_MSE)],argmin(pixel_MSE)

def predict_pixel_GMM_color_MSE_colorgraymix(pixel, gmm, mix_ratio=0.5):
    pixel_MSE_gray = ((rgb2gray(gmm.means_) - rgb2gray(pixel)) ** 2)
    pixel_MSE_color = ((gmm.means_ - pixel) ** 2).mean(axis=1)
    pixel_MSE = pixel_MSE_gray*mix_ratio + pixel_MSE_color*(1-mix_ratio)
    return gmm.means_[argmin(pixel_MSE)],argmin(pixel_MSE)

def create_image_from_GMM(rgb_image_arr,gmm,downsample_factor = 0.1, show_original = False, gray = True):

    rgb_image_arr_downsampled = misc.imresize(rgb_image_arr,downsample_factor)

    
    if show_original == True:
        plt.figure(figsize=(10,10))
        imshow(rgb_image_arr_downsampled)

    gmm_image_prediction = empty_like(rgb_image_arr_downsampled)

    predict_pixel_GMM_color = predict_pixel_GMM_color_MSE
    if gray == True:
        predict_pixel_GMM_color = predict_pixel_GMM_color_MSE_gray
    
    for x in range(rgb_image_arr_downsampled.shape[0]):
        for y in range(rgb_image_arr_downsampled.shape[1]):
            gmm_image_prediction[x,y] = predict_pixel_GMM_color(rgb_image_arr_downsampled[x,y],gmm)[0]

    plt.figure(figsize=(10,10))
    imshow(gmm_image_prediction)
    
    
    
    
    
def random_walk_on_GMM_cluster(rgb_image_arr,gmm,cluster_xy,downsample_factor = 0.1, show_original = False, gray = True):
    
    rgb_image_arr_downsampled = misc.imresize(rgb_image_arr,downsample_factor)
    
    
    if show_original == True:
        plt.figure(figsize=(10,10))
        imshow(rgb_image_arr_downsampled)
    
    predict_pixel_GMM_color = predict_pixel_GMM_color_MSE
    if gray == True:
        predict_pixel_GMM_color = predict_pixel_GMM_color_MSE_gray
    
    closest_cluster_color = predict_pixel_GMM_color(rgb_image_arr_downsampled[cluster_xy[0],cluster_xy[1]],gmm)
    
    # random walk steps 
#    closest_cluster_color_modified = closest_cluster_color + 
        
    
    for x in range(rgb_image_arr_downsampled.shape[0]):
        for y in range(rgb_image_arr_downsampled.shape[1]):
            gmm_image_prediction[x,y] = predict_pixel_GMM_color(rgb_image_arr_downsampled[x,y],gmm)
    
    plt.figure(figsize=(10,10))
    imshow(gmm_image_prediction)
    
    
