# **Behavioral Cloning** 

## Project Description
In this project, I used a neural network to clone car driving behavior. It is a supervised regression problem between the car steering angles and the road images in front of a car.
Those images were taken from three different camera angles (center, left, right) of the Car.
The network is based on The NVIDIA model and as image processing is involved, the model is using convolutional layers for automated feature engineering.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Train the Model

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.

It is a deep convolution network which works well with supervised image classification / regression problems. As the NVIDIA model is well documented, I was able to focus how to adjust the training images to produce the best result with some adjustments to the model to avoid overfitting and adding non-linearity to improve the prediction.

I've added the following adjustments to the model.

** I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
** I've added an additional dropout layer to avoid overfitting after the convolution layers.
** I've also included ELU for activation function for every layer except for the output layer to introduce non-linearity.

####NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
  
   *The convolution layers are meant to handle feature engineering the fully connected layer for predicting the steering angle.
   *dropout avoids overfitting
   *ELU(Exponential linear unit) function takes care of the Vanishing gradient problem. 

#### Model Structure
```sh
def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
```    
#### 4. Data Preprocessing

**the images are cropped so that the model won’t be trained with the sky and the car front parts
**the images are resized to 66x200 (3 YUV channels) as per NVIDIA model
**the images are normalized (image data divided by 127.5 and subtracted 1.0). As stated in the Model Architecture section, this is to avoid saturation and make gradients work better)

# Model Training

### Augumentation of Image 
For training, I used the following augumentation technique along with Python generator to generate unlimited number of images:

** Randomly choose right, left or center images.
** For left image, steering angle is adjusted by +0.2
** For right image, steering angle is adjusted by -0.2
** Randomly flip image left/right
** Randomly translate image horizontally with steering angle adjustment (0.002 per pixel shift)
** Randomly translate image virtically
** Randomly added shadows
** Randomly altering image brightness (lighter or darker)

Using the left/right images is useful to train the recovery driving scenario. The horizontal translation is useful for difficult curve handling (i.e. the one after the bridge).

#### Training, Validation and Test
I splitted the images into train and validation set in order to measure the performance at every epoch. Testing was done using the simulator.

As for training,

** I used mean squared error for the loss function to measure how close the model predicts to the given steering angle for each image.
** I used Adam optimizer for optimization with learning rate of 1.0e-4 which is smaller than the default of 1.0e-3. The default value was too big and made the validation loss stop improving too soon.
** I used ModelCheckpoint from Keras to save the model only if the validation loss is improved which is checked for every epoch.

## The Lake Side Track
As there can be unlimited number of images augmented, I set the samples per epoch to 20,000. I tried from 1 to 200 epochs but I found 5-10 epochs is good enough to produce a well trained model for the lake side track. The batch size of 40 was chosen as that is the maximum size which does not cause out of memory error on my Mac with NVIDIA GeForce GT 650M 1024 MB.

[image1]: ./run1/2019_02_05_14_37_50_122.jpg "Center"
[image2]: ./run1/2019_02_05_14_37_51_510.jpg
[image3]: ./run1/2019_02_05_14_38_14_842.jpg

#### 3. Outout

The model can drive the course without bumping into the side ways.

Here's a [link to my video result](./Behavioral Cloning Video.mp4)

## Credit
NVIDIA model: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
