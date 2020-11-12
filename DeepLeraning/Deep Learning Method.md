# Abstract

# Objectives
Autonomous vehicles (AVs) are expected to dramatically redefine the future of transportation. However, there are still significant engineering challenges to be solved before one can fully realize the benefits of self-driving cars. One such challenge is building models that reliably predict the movement of traffic agents around the AV, such as cars, cyclists, and pedestrians. The ridesharing company Lyft released a challenge to predict the motion of traffic agents (e.g. pedestrians, vehicles, bikes, etc.). In this competition, you'll have access to the largest Prediction Dataset ever released to train and test your models.
The goal of this shared code is to predict the trajectories of traffic agents. Amulti-modal ones generating three hypotheses (mode of transportation).
# Data (Input)
The Lyft Motion Prediction for Autonomous Vehicles competition is fairly unique, data-wise. In it, a very large amount of data is provided, which can be used in many different ways. Reading the data is also complex.
The data is packaged in .zarr files. These are loaded using the zarr Python module, and are also loaded natively by l5kit. Each .zarr file contains a set of:
*scenes: driving episodes acquired from a given vehicle.
*frames: snapshots in time of the pose of the vehicle.
*agents: a generic entity captured by the vehicle's sensors. Note that only 4 of the 17 possible agent label_probabilities are present in this dataset.
*agents_mask: a mask that (for train and validation) masks out objects that aren't useful for training. In test, the mask (provided in files as mask.npz) masks out any test object for which predictions are NOT required.
traffic_light_faces: traffic light information.


# Method
## Model Archeticture
## Optimizer
## Input shape
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
