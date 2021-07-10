### 2nd Affective Behavior Analysis in-the-wild (ABAW2) Competition

Source code for Team Keegs entry to the ABAW2 competition.

The file wild.py consists of the training loop and wild_dataset is the dataset class required by pytorch. The pre-trained model used was taken from https://github.com/Open-Debin/Emotion-FAN created by Menge et al. (https://arxiv.org/abs/1907.00193).

The images and labels provided by the competition hosts was read and stored as a pickle file. Additional experiments were done using dcgan implementation of pytorch, however, that is not included in the final paper.
