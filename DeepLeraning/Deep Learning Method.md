# Abstract

# Objectives
Autonomous vehicles (AVs) are expected to dramatically redefine the future of transportation. However, there are still significant engineering challenges to be solved before one can fully realize the benefits of self-driving cars. One such challenge is building models that reliably predict the movement of traffic agents around the AV, such as cars, cyclists, and pedestrians. The ridesharing company Lyft released a challenge to predict the motion of traffic agents (e.g. pedestrians, vehicles, bikes, etc.). In this competition, you'll have access to the largest Prediction Dataset ever released to train and test your models.
The goal of this shared code is to predict the trajectories of traffic agents. Amulti-modal ones generating three hypotheses (mode of transportation). 
_We are predicting the motion of the objects in a given scene._
# Data (Input)
The Lyft Motion Prediction for Autonomous Vehicles competition is fairly unique, data-wise. In it, a very large amount of data is provided, which can be used in many different ways. Reading the data is also complex.
The data is packaged in .zarr files. These are loaded using the zarr Python module, and are also loaded natively by l5kit. Each .zarr file contains a set of:
* scenes: driving episodes acquired from a given vehicle.
* frames: snapshots in time of the pose of the vehicle.
* agents: a generic entity captured by the vehicle's sensors. Note that only 4 of the 17 possible agent label_probabilities are present in this dataset.
* agents_mask: a mask that (for train and validation) masks out objects that aren't useful for training. In test, the mask (provided in files as mask.npz) masks out any test object for which predictions are NOT required.
traffic_light_faces: traffic light information.
For detailed information on data format and how to deal with it https://github.com/lyft/l5kit/blob/master/data_format.md
For detailed license information plese refer to https://self-driving.lyft.com/level5/prediction/
For further information of the data collection please refer to https://arxiv.org/pdf/2006.14480.pdf
# Method
## Model Archeticture
Now to the fun part! 
The objective of this project is quite intersting as it is combinig between two tasks 1) training the model to learn from images/frames 2)
and training the model to understand the temoral correlation between sequence of frames (scene). The below figure summarize the model 1's network archeticure.
* Branch(1)
So I decided to start the model by a pretrained model namely ResNet18 at branch (1). This will help the model to have a good inititalization. Hopefully by the end of this step the model starts to understand what is going on each frame (notice that the model did not yet relate between different frames). The output from ResNet model is fed into a long short-term memory (LSTM). This should help the model to account for the temporal correlation in the data (e.g. how previous frames affect future frames).
* Branch(2)
I noticed that branch (1) is so deep so I decided to add the same input through a shallower link to an LSTM. This is mainly to help the model learn directly from the input before it gets too complicated.

The output from the LSTM layers in branch (1&2) are concatenated and fed into a long short-term memory(LSTM) network. The LSTM output is then used as input as regressor to predict the agents' location (x,y) in the next 50 frmaes given that it might be one of three modes, so we are predicting 2(location ID) x 50(number of future steps) x 3(number of modes).
## implementation
All models are trained using the Adam optimizer with an initial learning rate of 0.001, without weight decay, and momentumwith β1 = 0.9, and β2 = 0.9999. Minibatch size is 16 limited by GPU memory and CPU (data decompression). The image set is of dimensions 224x224 using ResNet18.
## Loss function
We calculate the negative log-likelihood of the ground truth data given the multi-modal predictions. Let us take a closer look at this. Assume, ground truth positions of a sample trajectory are

![Eq1](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq1.PNG)

In addition, we predict confidences c of these K hypotheses. We assume the ground truth positions to be modeled by a mixture of multi-dimensional independent Normal distributions over time, yielding the likelihood

![Eq2](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq2.PNG)

which results in the loss

![Eq3](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq3.PNG)

For further informatyion on the used loss funcation please refer to https://github.com/lyft/l5kit/blob/master/competition.md
# Suggestions

# Next step

# Limitations
History length, raster size
