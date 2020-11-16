# Objectives
Autonomous vehicles (AVs) are expected to dramatically redefine the future of transportation. However, there are still significant engineering challenges to be solved before one can fully realize the benefits of self-driving cars. One such challenge is building models that reliably predict the movement of traffic agents around the AV, such as cars, cyclists, and pedestrians. The ridesharing company Lyft released a challenge to predict the motion of traffic agents (e.g. pedestrians, vehicles, bikes, etc.). In this competition, you'll have access to the largest Prediction Dataset ever released to train and test your models.
The goal of this shared code is to predict the trajectories of traffic agents. A multi-modal generating three hypotheses (mode of transportation). 
_We are predicting the motion of the objects in a given scene._
# Data (Input)
The Lyft Motion Prediction for Autonomous Vehicles competition is fairly unique, data-wise. In it, a very large amount of data is provided, which can be used in many different ways. Reading the data is also complex.
The data is packaged in .zarr files. These are loaded using the zarr Python module and are also loaded natively by l5kit. Each .zarr file contains a set of:
* scenes: driving episodes acquired from a given vehicle.
* frames: snapshots in time of the pose of the vehicle.
* agents: a generic entity captured by the vehicle's sensors. Note that only 4 of the 17 possible agent label_probabilities are present in this dataset.
* agents_mask: a mask that (for train and validation) masks out objects that aren't useful for training. In the test, the mask (provided in files as a mask.npz) masks out any test object for which predictions are NOT required.
traffic_light_faces: traffic light information.

For detailed information on data format and how to deal with it https://github.com/lyft/l5kit/blob/master/data_format.md

For detailed license information please refer to https://self-driving.lyft.com/level5/prediction/

For further information on the data collection please refer to https://arxiv.org/pdf/2006.14480.pdf

# Method
## Model Architecture
Now to the fun part! 
The objective of this project is quite interesting as it is combining two tasks 1) training the model to learn from images/frames 2)
and training the model to understand the temporal correlation between the sequence of frames (scene). The below figure summarizes model 1's network architecture.
![Model101](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/Model111.PNG)
* Branch(1)
So I decided to start the model with a pre-trained model namely ResNet18 at the branch (1). This will help the model to have a good initialization. Hopefully, by the end of this step, the model starts to understand what is going on in each frame (notice that the model did not yet relate between different frames). The output from the ResNet model is fed into a long short-term memory (LSTM). This should help the model to account for the temporal correlation in the data (e.g. how previous frames affect future frames).
* Branch(2)
I noticed that branch (1) is so deep so I decided to add the same input through a shallower link to an LSTM. This is mainly to help the model learn directly from the input before it gets too complicated.

The output from the LSTM layers in the branch (1&2) are concatenated and fed into a long short-term memory(LSTM) network. The LSTM output is then used as input as a regressor to predict the agents' location (x,y) in the next 50 frames given that it might be one of three modes, so we are predicting 2(location ID) x 50(number of future steps) x 3(number of modes).
## implementation
All models are trained using Adam optimizer with an initial learning rate of 0.001, without weight decay, and momentum with β1 = 0.9, and β2 = 0.9999. Minibatch size is 16 limited by GPU memory and CPU (data decompression). The image set is of dimensions 224x224 using ResNet18.
## Loss function
We calculate the negative log-likelihood of the ground truth data given the multi-modal predictions. Let us take a closer look at this. Assume, ground truth positions of a sample trajectory are

![Eq1](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq1.PNG)

Also, I predict confidences c of these K hypotheses. I assume the ground truth positions to be modeled by a mixture of multi-dimensional independent Normal distributions over time, yielding the likelihood

![Eq2](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq2.PNG)

which results in the loss

![Eq3](https://github.com/MKamel1/Kaggle_Lyft/blob/master/DeepLeraning/images/eq3.PNG)

For further information on the used loss function please refer to https://github.com/lyft/l5kit/blob/master/competition.md
# Suggestions
* Different configuration(e.g. number of history frames, raster size) of the input should be tried in the future.
* Different pre-trained models could be adopted (Xception, ResNet34).
* For the temporal correlation, instead of the LSTM convolutional layers could have been used similar to WaveNet (Hint: 2 dimensions instead of one dimension)
* Endless Architecture could be adopted but a promising one is to try to feed the pre-trained model output and the final LSTM layer output to the fully connected layer. 
# Limitations
* Due to computational limitation the model is only trained on 0.5% of the data in other words only 0.5% x 1 epoch. I believe the model can behave much better if it at least trained on 1 epoch.
* Note that we should have monitored the loss on the validation dataset while training the model (to avoid overfitting the model), however, this was not adopted as data decompression is very expensive. To illustrate the data decompression for the training set while training is already the bottleneck in the training process. Also, I am not so worried about overfitting the data as we will not do many epochs preferably just one epoch and the data seems big for the objective.
