
# **Behavioral Cloning** 

**Behavioral Cloning Project**

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
* model.hdf5 containing a checkpoint for the model
* writeup_report.md or writeup_report.pdf summarizing the results
* run1.mp4 showing the video result.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

[//]: # (Image References)

[center_image]: data/IMG/center_2016_12_01_13_30_48_287.jpg "Center Image"
[left_image]: data/IMG/left_2016_12_01_13_30_48_287.jpg "Left Image"
[right_image]: data/IMG/right_2016_12_01_13_30_48_287.jpg "Right Image"


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes with dropout layers and fully connected layers as well. (model.py lines 59-73) 

The model is basically an implementation of this [research article](https://devblogs.nvidia.com/deep-learning-self-driving-cars/).  It consists of 5 convolutional layers and 4 full connected layers and a normalization layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 67). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56 and 57). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 74).

#### 4. Appropriate training data

For the training data I ended up using the dataset provided to me by Udacity and augmented it. The reason being, with my slow internet connectivity, I was unable to upload large training datasets from my local machine and unfortunately the online simulator working as intended.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make sure the car stayed at the center of the road.

My first step was to use a convolution neural network model similar to the LeNet architecture. This model did pretty good on traffic signal classification and it was also mentioned in the videos. Now, there are other good models as well which is also simple to use like ResNet, VGG, Inception etc. which too classifies images pretty well but I decided to stay with LeNet as I had already implemented it.

After going through some more videos, I found a mention of one of nVidia's paper on End-to-End Deep Learning model for Self driving cars. It was perfect for my scenario and I decided to implement that.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I wanted to train to for atleast 5 epochs to see how it performs. However, I found that it took quite a long time to train the model with my current approach. Also, the generator did not help much. After, the reviewer's comments to implement a better generator, I fiddled around some more and stumbled across another activation function called ["elu"](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf).

I implemented that and re-wrote my code to augment images on-the-fly and shuffle them with my new generator. The training time decreased exponentially. The current architecture is discussed below.

#### 2. Final Model Architecture

The final model architecture (model.py lines 59-73) consisted of a convolution neural network with the following layers and layer sizes:
| code line | Description  
| --- | ---
|  60 | Normalizes the image data via a lambda function
|  61 | Crops the image by 70 pixels on the top and 25 pixels on the bottom
|  62 | Convolution layer with the following arguments- filter: 5x5, depth: 24, stride: 2x2, Activation: elu.
|  63 | Convolution layer with the following arguments- filter: 5x5, depth: 36, stride: 2x2, Activation: elu.
|  64 | Convolution layer with the following arguments- filter: 5x5, depth: 48, stride: 2x2, Activation: elu.
|  65 | Convolution layer with the following arguments- filter: 3x3, depth: 64, Activation: elu.
|  66 | Convolution layer with the following arguments- filter: 3x3, depth: 64, Activation: elu.
|  67 | Uses a dropout layer with a 0.25 probability in order to not **over fit**.
|  68 | Flatten the matrix for the following dense/linear operations.
|  69 | Uses the fully connected dense layer to reduce the output dimension to 100. Activation function used: elu.
|  70 | Uses the fully connected dense layer to reduce the output dimension to 50. Activation function used: elu.
|  71 | Uses the fully connected dense layer to reduce the output dimension to 10. Activation function used: elu.
|  72 | Uses the fully connected dense layer to reduce the output dimension to 1.
|  73 | Use the Adam optimizer with a MSE loss function. The learning rate was **not tuned manually**.

#### 3. Creation of the Training Set & Training Process

The dataset provided by Udacity contained enough data to properly train and generalize the model. Combined with the flipping of images, the model seemed to have enough data to generalize even for track 2. The dataset contained 3 parts of an images from three camera angles (centre, left and right side of the road through the car) as shown below:

![Center Image][center_image]
![Left Image][left_image]
![Right Image][right_image]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I augmented the image using the cv2.flip method and some corrections to steerings. These were done on-the-fly using a generator to speed up the training process. I used an adam optimizer so that manually training the learning rate wasn't necessary.
