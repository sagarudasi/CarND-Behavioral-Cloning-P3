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

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

Main project files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network (https://drive.google.com/open?id=0B8ZLsM_0WsDSUG5uemtvYXVDQ00)
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

Apart from these layers, few more layers were added on top of the model for cropping and normalization.

Following is the model visualization created using keras.utils.vis_utils -
![alt text][image1]

I used batch size of 128 to train the network for over 10 iteration which reduced the validation loss to less ~ 0.018.

There was no need to define learning rate as Adam optimizer was involved and loss function used was "Mean squared error".

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