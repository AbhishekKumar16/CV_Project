# CV_Project
CS 763 Computer Vision Project


Implementation of Re3 in PyTorch: Real-Time Recurrent Regression Networks for
Visual Tracking of Generic Objects
Based on the <a href = "https://arxiv.org/pdf/1705.06368.pdf"> this paper</a> by Daniel Gordon, Ali Farhadi, and Dieter Fox 

Robust object tracking is a fundamental problem in Computer Vision. It plays an important role in several areas of robotics. Re3 provides a lightweight model to track objects robustly and also incorporate temporal information into its model. It handles temporary occlusion too.

Consists of convolutional layers to embed the object appearance, recurrent layers to remember appearance and motion information, and a regression layer to output the location of the object.

The way this is carried out is by the following steps:-

Object appearance embedding
<ul>
<li>Learn the feature extraction directly by using a convolutional pipeline that can be trained fully end-to-end on a large amount of data</li>
<li>At each frame, the network is feeded with a pair of crops from the image sequence. The first crop is centered at the object’s location in the previous image, whereas the second crop is in the same location, but in the current image. The crops are each padded to be twice the size of the object’s bounding box to provide the network with context</li>
<li>The crops are warped to be 227 × 227 pixels before being input into the network</li>
<li>Skip connections are used when spatial resolution decreases to give the network a richer appearance model</li>
<li>The skip connections are each fed through their own 1 × 1 × C convolutional layers where C is chosen to be less than the number of input channels</li>

</ul>


The two crops
<figure>
<img src="/images_readme/cv1.png" />
<figcaption>Visualization of images with bounding box at frames i and i-1</figcaption>
</figure>


<figure>
<img src="/images_readme/cv2.png" />
<figcaption>Image crops fed to the network</figcaption>
</figure>


For training I have specifically used the ALOV300++ dataset. This dataset can be downloaded from <a href="http://alov300pp.joomlafree.it/">this link.</a> Although the actual paper suggests using ImageNet video dataset as well and to further make synthetic datapoints from the ImageNet video dataset, I sticked to using the ALOV300++ dataset due to a constraint on the resources available and the time for training. A lot of code for preprocessing etc has been taken from the python version of <a href="https://github.com/amoudgl/pygoturn">goturn</a>(as the feeding of crops etc were similar)

In the ALOV dataset downloaded, keep the ground truth values and the actual data into a folder named 'alov' just outside the folder of this repository. Let the name of the directory for the actual video data frames be 'imagedata++' and the corresponding annotations be 'alov300++\_rectangleAnnotation_full'.
I also took out the last entires from each of the directories of imagedata++ to construct the test set. Place the test set in the imagedata++ directory.

The code can then be directly run using python3 trainModel.py
To test the code, in the file testModel.py, change model_weights on line 26 to the file name of the saved model.

I could not complete the network's training using the entire dataset(due to it requiring a lot of computational time and resources) and hence the trained model is unavailable. So to test the network, it has to be trained first using the command mentioned above and then used for testing.
