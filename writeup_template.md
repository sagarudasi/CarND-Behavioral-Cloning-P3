#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/recovery_1.jpg "Recovery Image"
[image3]: ./examples/recovery_2.jpg "Recovery Image"
[image4]: ./examples/recovery_3.jpg "Recovery Image"
[image5]: ./examples/offtrack_1.jpg "Off track 1"
[image6]: ./examples/offtrack_2.jpg "Off track 2"
[image7]: ./examples/cropped.jpg "Cropped image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

Main project files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (https://drive.google.com/open?id=0B8ZLsM_0WsDSNDlJU2k4bGJvNzg)
* writeup_report.md or writeup_report.pdf summarizing the results

Additional files: (Uploaded on Google Drive): https://drive.google.com/open?id=0B8ZLsM_0WsDSUG5uemtvYXVDQ00 
* Compressed images dataset used for training
* Video recording of autonomous mode driving  

####2. Submission includes functional code

Following are the steps which can be followed to generate the model from the dataset 
* Download the compressed images dataset and extract it in the same directory where python files are kept 
* Modify the model.py for the basepath (if necessary)
* Run the following command to generate the model

```sh 
python model.py
```

This should train the network and save a file named model.h5 to the same working directory.

Now for testing the model, launch simulator and select autonomous mode and execute the below command 
```sh
python drive.py model.h5
```

For recording the run use the below alternative command 
```sh 
python drive.py model.h5 recordingdirectory
```

This will generate JPEG images of the run.

####3. Submission code is usable and readable

Python file to generate the model is model.py in which I have provided comments at necessary places. 
Comments are also provided at the start of following sections - Deciding hyperparameters, reading dataset, pre-processing, convolution layers, dense layers etc.

Script to drive the file i.e. drive.py is not modified and can be used as is to run the car in simulator.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model was based on NVIDIA's model and it consists of the following layes -

##### Convolutional layers 
1. 5x5 Conv2D with 24 output channels 
2. MaxPooling 2x2
3. 5x5 Conv2D with 36 output channels 
4. MaxPooling 2x2
5. 5x5 Conv2D with 48 output channels 
6. MaxPooling 2x2
7. 3x3 Conv2D with 64 output channels 
8. 3x3 Conv2D with 64 output channels

##### Fully connected layers
1. Dense layer with 1164 neurons 
2. Dense layer with 100 neurons
3. Dense layer with 50 neurons
4. Dense layer with 10 neurons
5. Output layer with 1 neuron 


All the layers except the last one uses RELU for activation. Last layer uses linear activation function for output.

####2. Attempts to reduce overfitting in the model

I trained the model for 20 iterations and saw the model was overfitting and the difference between the validation set and train set was increasing.

I decided to lower down the iterations to 10 in order to avoid overfitting of data.

Another way of avoiding overfitting is adding dropout layers. 

Dropout layers can be added using the following function of keras 

```python 
dropout(rate, noise_shape=None, seed=None)
```

Initially I added dropout to the first fully connected layer. However, with the amount of data recording and iterations used, I saw the dropout not contributing significantly hence removed those layers.

####3. Model parameter tuning

Adam optimizer was used as an optimizer function. Adam optimizer uses decaying learning rate which results in smooth and consistent train validation loss during the last cycles of training.
Another optimizer which could have been used is SGD Stochastic gradient descent optimizer. Keras function for the same is SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

Batch size of 128 was used in-order to let the network converge better at once.

Also, as described in the above section, the number of epochs used were 10.
 

####4. Appropriate training data

Training scenarios included driving in center of lane, driving in the opposite direction and recovery scenarios.
It was also taken care that while recording the training data, there are no mistakes like driving off the road which may result in incorrect learning.

Training data was also pre-processed by cropping ROI (region of interest) and then it was used for training.

Following the final cropped image used - 

![Cropped image][image7]

###Model Architecture and Training Strategy

####1. Solution Design Approach

The input to the network was going to be a 2D image.

The output expected is a steering angle which can be used by car to steer it on the road.

So we need to design a network that properly maps the two dimensional color image to steering angle which is in float.

The best solution which can be deployed for 2D images is by applying convolution and then boil down the data to the required value.

Hence, convolution layers were used.

Also in the process of boiling the data down through the network, since we need one value at the end, we need to flatten the network and gradually get the desired output.

For this purpose, dense layers were added to the network.

####2. Final Model Architecture

My final model can be found from line number 75 to 101 in model.py. Basically its the NVIDIA network modified a little bit by adding few layers.

Apart from NVIDIA network layers, few more layers were added on top of the model for cropping and normalization.

```python
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(orow, ocol, input_channels), output_shape=(orow, ocol, input_channels)))
model.add(Reshape((input_channels, orow, ocol)))
model.add(Cropping2D(cropping=((ctop,cbottom), (cleft,cright)), input_shape=(input_channels, orow, ocol)))
```

So the final model architecture consisted of 5 convolution layers and 5 dense layers along with 2 pre-processing layers.

Following is the model visualization created using keras.utils.vis_utils -
![Model visualization][image1]

####3. Creation of the Training Set & Training Process

To create the training set, I recorded images by driving two laps in one direction and two laps in opposite directions.

Then I recorded couple of recovery images as following when the car was driven back to center from left and right lane lines.

Below are the example images of recovering from the extreme left to center of track-
![Recovery image][image2] 
![Recovery image][image3] 
![Recovery image][image4]

It was important to record recovery images on multiple locations where the lane lines were different (Two yellow, white and red stripes etc).

Few more normal laps of center driving and few more recovery images were captured in order to make sure the validation set is also a proper mix of images.

The final trained model works fine as seen in the recorded mp4 video. However, at some points it tries drive out of the track but recovers back to the center.
Following are the two images where the car tries to go offtrack -
![off track image][image5] 
![off track image][image6]

I think couple of more recovery training and specially training recovery only from yellow lane lines (And not from the white lines) may help solve this problem.
Also using left and right camera images will provide better stability. 
