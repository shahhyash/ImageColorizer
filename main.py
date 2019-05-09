from os import listdir
from os.path import join, exists
from os import makedirs, remove

import numpy as np
import random

from scipy import signal
import cv2
from PIL import Image

TRAINING_INPUT_DIR=join("training_data", "bw")
TRAINING_OUTPUT_DIR=join("training_data", "color")
TEST_IMAGES_DIR="test_images"
WEIGHTS_DIR="weights"

#  Given a value x, it fetches the sigmoid value at that point
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given a value x, it fetches the derivative of the sigmoid function at that point
def d_sigmoid(x):
    return x * (1 - x)

# Using coordinates to a pixel in the image, it creates and returns a grid of 3x3 pixel patch with it's surrounding pixels
def fetchPixelPatch(img, px_x, px_y):
    imgNeighbors = [ [], [], [], [], [], [], [], [], [] ]
    imgNeighbors[0] = [img[px_x - 1][px_y - 1]]
    imgNeighbors[1] = [img[px_x - 1][px_y]]
    imgNeighbors[2] = [img[px_x - 1][px_y + 1]]
    imgNeighbors[3] = [img[px_x][px_y - 1]]
    imgNeighbors[4] = [img[px_x][px_y]]
    imgNeighbors[5] = [img[px_x][px_y + 1]]
    imgNeighbors[6] = [img[px_x + 1][px_y - 1]]
    imgNeighbors[7] = [img[px_x + 1][px_y]]
    imgNeighbors[8] = [img[px_x + 1][px_y + 1]]

    return np.array(imgNeighbors)

# Given a training image file name, this function fetches all required data and returns it
def fetchTrainingDataForImage(file_name):
    grayscale_img = cv2.imread(join(TRAINING_INPUT_DIR, file_name), 0)
    color_img = cv2.imread(join(TRAINING_OUTPUT_DIR, file_name))
    blue_filter, green_filter, red_filter = cv2.split(color_img)
    return np.array(grayscale_img), np.array(blue_filter), np.array(green_filter), np.array(red_filter)

# Trains the network and stores its respective weights for each layer in a separate file.
# This specific neural network makes use of two hidden layers of 255 nodes each
def trainNetwork():
    # Begin by resetting numpy.random
    np.random.seed(1)

    # Next, we initialize the weights for each layer to random values using numpy's random function
    # Take note of the fact that our dimensions here can help you visualize our network as we start with 9 inputs, moving onto hidden layers of 255 nodes, and finally to 3 outputs representing RGB values.
    # Also, we found that numpy's random function only initializes all weights to a positive value between 0-1. 
    # In order to also start with a few negative weights to make it easier to train the model on more accurate results, we multiple this value by 2 and subtract by 1
    layer1_weights = 2 * np.random.random((255,9)) - 1
    layer2_weights = 2 * np.random.random((255,255)) - 1
    output_weights = 2 * np.random.random((3,255)) - 1
    
    # Next, we will open our training_data directory and fetch all images that we'll use to train this network
    for picture in listdir(TRAINING_OUTPUT_DIR):
        print("Training network for picture: ", picture)

        # Fetch grayscale image, and the inidividual color filters for the picture
        gray, blue, green, red = fetchTrainingDataForImage(picture)
        
        # Pad the image with a border of white pixels around the edge
        padded_image = np.pad(gray, 1, 'edge')

        # Here the epoch for training the network on each picture is set to 3 because it provided the best results
        epochs = 3    
        for x in range(epochs):
            # Append all possible pixel coordinates for the image into a list. 
            # We use this method as it allows us to eliminate a certain level of bias by randomly picking pixels
            coords = []
            for px_x in range(len(gray)):
                for px_y in range(len(gray)):
                    coords.append((px_x, px_y))

            while len(coords) > 0:
                # Pick a random pixel from the list of coordinates
                random_set = random.randint(0, len(coords) - 1)
                i, j = coords.pop(random_set)
                patch = fetchPixelPatch(padded_image, i, j) / 255

                # fetch rgb values of colored pixel
                center_pixel = np.array([[red[i][j]], [green[i][j]], [blue[i][j]]]) / 255
                
                # forward propagate once for this pixel data on the current set of weights
                layer1_result = sigmoid(np.dot(layer1_weights, patch))
                layer2_result = sigmoid(np.dot(layer2_weights, layer1_result))
                output = sigmoid(np.dot(output_weights, layer2_result))
                
                # Perform back propogation to compute the changes to the weights for each layer
                output_error = center_pixel - output
                d_output = output_error * d_sigmoid(output)

                layer2_error = output_weights.T.dot(d_output)
                d_layer2 = layer2_error * d_sigmoid(layer2_result)
                
                layer1_error = layer2_weights.T.dot(d_layer2)
                d_layer1 = layer1_error * d_sigmoid(layer1_result)
                
                # Update the global weights
                output_weights += d_output.dot(layer2_result.T)
                layer2_weights += d_layer2.dot(layer1_result.T)
                layer1_weights += d_layer1.dot(patch.T)

                # print('Error at pixel ', (i,j), ' = ', np.mean(np.abs(error)))
            print('Error Value for picture', picture, 'for epoch', x, '=', np.mean(np.abs(output_error)))
    
    # make sure we're working with an empty directory
    if not exists(WEIGHTS_DIR):
        makedirs(WEIGHTS_DIR)
    else:
        for file in listdir(WEIGHTS_DIR):
            remove(join(WEIGHTS_DIR, file))

    # Save the weights to a file to avoid recomputing them on each run
    np.savetxt(join(WEIGHTS_DIR, "layer1_weights"), layer1_weights)
    np.savetxt(join(WEIGHTS_DIR, "layer2_weights"), layer2_weights)
    np.savetxt(join(WEIGHTS_DIR, "layer3_weights"), output_weights)
        
    return layer1_weights, layer2_weights, output_weights

# Given a set of weights and the image we should be able to recreate the colored image
def colorImage(grayscale_img, layer1_weights, layer2_weights, layer3_weights):
    # Set up storage for pixel values for each color filter
    B_output = np.zeros((256, 256), dtype='uint8')
    G_output = np.zeros((256, 256), dtype='uint8')
    R_output = np.zeros((256, 256), dtype='uint8')

    # Pad image around border to compute all values
    padded_image = np.pad(grayscale_img, 1, 'edge')
    for i in range(len(grayscale_img)):
        for j in range(len(grayscale_img)):
            # Fetch 3x3 pixel patch around current pixel
            patch = fetchPixelPatch(padded_image, i, j) / 255

            # Compute results layer by layer until we get to output        
            layer1_result = sigmoid(np.dot(layer1_weights, patch))
            layer2_result = sigmoid(np.dot(layer2_weights, layer1_result))
            output = sigmoid(np.dot(layer3_weights, layer2_result))
           
            # Store RGB values in respective filter and pixel
            R_output[i][j] = output[0][0] * 255
            G_output[i][j] = output[1][0] * 255
            B_output[i][j] = output[2][0] * 255

    # Using opencv, merge all three color filters into one image and display it
    mergedChannels = cv2.merge((B_output, G_output, R_output))
    cv2.imshow("output", mergedChannels)
    cv2.waitKey(0)
            
if __name__ == '__main__':
    
    # Write the entire name of the file you'd like to test with available in the test_images directory
    # The test_images directory does not have any pictures that were explicitly trained with in the model
    test_image = 'land604.jpg'
    test_image = cv2.imread(join(TEST_IMAGES_DIR, test_image), 0)
    
    # Uncomment the line below if you'd like to retrain the network
    # l1w, l2w, l3w = trainNetwork()

    # Uncomment the next three lines if you'd like to use existing weights
    l1w = np.loadtxt(join(WEIGHTS_DIR, "layer1_weights"))
    l2w = np.loadtxt(join(WEIGHTS_DIR, "layer2_weights"))
    l3w = np.loadtxt(join(WEIGHTS_DIR, "layer3_weights"))

    colorImage(np.array(test_image), l1w, l2w, l3w)