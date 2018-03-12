# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/NVIDIA_driving_model.png
[image2]: ./examples/sharp_turn1.jpg "Sharp turn 1"
[image3]: ./examples/sharp_turn2.jpg "Sharp turn 2"
[image4]: ./examples/recovery1.jpg "Recovery Image"
[image5]: ./examples/recovery2.jpg	"Recovery Image"
[image6]: ./examples/recovery3.jpg "Recovery Image"
[image7]: ./examples/recovery4.jpg "Recovery Image"
[image8]: ./examples/example_left.jpg "Left sample"
[image9]: ./examples/example_center.jpg "Center sample"
[image10]: ./examples/example_right.jpg "Right sample"
[image11]: ./examples/sample.jpg "Sample"
[image12]: ./examples/sample_flipped.jpg "Flipped"

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

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is inspired by NVIDIA's ["End to End Learning for Self-Driving Cars"](https://arxiv.org/abs/1604.07316). An illustration of the model is provided below:

![Model architecture][image1]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 153-156). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 178).

#### 4. Appropriate training data

Two datasets were chosen, the provided one from Udacity and another one kindly shared by [Ryan Moore](https://carnd.slack.com/archives/C2HQV18F2/p1488178009024678). The second dataset is captured on track 1 and contains two subsets, one with normal driving conditions and another with recovery condition. I have not used the normal driving data from this dataset, only the recovery set. Along with Udacity data, this forms my complete dataset.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Based on various feedback and suggestions I found on the discussion forums, I decided to start directly with the NVIDIA end-to-end model described above.

In the beginning I started by training the model on some of the data I collected with simulator. I only used the center images from the data. The data proved insufficient as the model quickly overfit in less than 10 epochs. The test of the model on the simulator showed that it had difficulty negotiating turns.

For my second attempt, I replaced my data with the Udacity data and trained the model for several epochs (~70 epochs) over different runs. Different runs meant that each time a slightly different training set was seen by the model due to random shuffling as described below. This did lead to much improved results in the simulator, but the car still could not negotiate sharp turns like the following:

![sharp turn 1][image2]                              ![sharp turn 2][image3] 

In such conditions, the car would go over the edge of the road and get stuck there. This pointed to two things:

* The data for sharp turns was insufficient 
* The model had not learned on how to recover from a wrong trajectory. 

I noticed that output steering angles remained very low. The model had not learned to do this because there was insufficient data with large turn angles. 

To correct these issues, I decided to fine tune the model with the recovery data mentioned above. This lead to improved performance on turns. The car was successful in the negotiating the first sharp turn shown above, but not the other. In the second sharp turn shown above, the car went over the edge of the road, although it did appear that the car was attempting to recover by driving at an angle.

My third and final attempt was to make use of the left and right camera images in addition to the center image. For these images, I also offset the steering angle with a factor of 0.2. This provided enough data with large angles to balance the dataset.

After this augmentation, the model worked quite well on sharp turns. The driving was a little wonky, but the car stayed within the lane and negotiated the turns well enough.

#### 2. Final Model Architecture

My final model was similar to the original model shown above, except that I added a Cropping layer, cropping out 50 pixels from the top and 20 pixels from the bottom. 

ReLU is used as the activation functions for all layers.

The implementation of the model can be found in function `createModel` (model.py lines 112-140)

#### 3. Creation of the Training Set & Training Process

The dataset two distinct behaviors that the car can emulate, normal driving and recovery driving. For each frame, along with the steering angle used, three images are recorded, each from cameras mounted on the left, center and right sides of the dash respectively. An example is shown below:

|      Left       |      Center       |       Right       |
| :-------------: | :---------------: | :---------------: |
| ![left][image8] | ![center][image9] | ![right][image10] |

The steering angles derived from left and right images were offset by a factor of 0.2 to maintain centrality. 

Along with data from normal (i.e. lane center driving), recovery data was also used. This was captured by recording data when the car starts recovering from the edge of the road to the center. An example is shown below:

|       Frame 1       |       Frame 2       |       Frame 3       |       Frame 4       |
| :-----------------: | :-----------------: | :-----------------: | :-----------------: |
| ![recovery][image4] | ![recovery][image5] | ![recovery][image6] | ![recovery][image7] |

All in all, this data had 28323 samples. This data is then randomly shuffled and split into training and validation sets, with validation comprising 20% of total data. While training, I also augment the data for each batch by flipping images along the vertical. This is done only if the steering angle for the corresponding image is greater than 0.1 as anything less than one signifies a straight road and flipping that won't make any noticeable difference. An example of this augmentation is shown below:

|     Original     |     Flipped      |
| :--------------: | :--------------: |
| ![orig][image11] | ![flip][image12] |

The model was trained for only 10 epochs, yet it produced some good results in the simulator.



For some future implementation ideas, I plan to modify the model to a three column Siamese network that can simultaneously accept all three images (left, center and right). Additionally, I also want to explore if data from previous frames can be used to predict the values of the current frame.