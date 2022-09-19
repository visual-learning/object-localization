# Assignment 1: Weakly Supervised Object Localization

- [Visual Learning and Recognition (16-824) Fall 2022](https://visual-learning.cs.cmu.edu/)
- Updated by: [Anirudh Chakravarthy](https://anirudh-chakravarthy.github.io/) and [Sai Shruthi Balaji](https://www.linkedin.com/in/sai-shruthi-balaji/)
- Created by : [Senthil Purushwalkam](http://www.cs.cmu.edu/~spurushw/)
- TAs: [Anirudh Chakravarthy](https://anirudh-chakravarthy.github.io/), [Sai Shruthi Balaji](https://www.linkedin.com/in/sai-shruthi-balaji/), [Vanshaj Chowdhary](https://www.linkedin.com/in/vanshajchowdhary/), and [Nikos Gkanatsios](https://nickgkan.github.io/).

- We will be keeping an updated FAQ on piazza. Please check the FAQ post before posting a question.
- Due date: Oct 3rd, 2022 at 11:59pm EST.
- Total points: 100

## Introduction
In this assignment, we will learn to train object detectors with only image-level annotations and no bounding box annotations! First, in task 1, we will use classification models and examine their backbone features for object localization cues. In task 2, we will train object detectors in the *weakly supervised* setting, which means we're going to train an object detector without bounding box annotations!

When we use a classification network like AlexNet, it is trained using a classification loss (cross-entropy). Therefore, in order to minimize this loss function, the network maximizes the likelihood for a given class. Since CNNs preserve spatial locality, this means that the model implicitly learns to produce high activations in the feature map around the regions where an object is present. We will use this property to approximately localize the object in the image. This is called a weakly-supervised paradigm: supervised because we have image-level classification labels, but weak since we don't have ground-truth bounding boxes.

We will use the [PyTorch](pytorch.org) framework to design our models, train and test them. We will also be using [Weights and Biases](https://wandb.ai/site) for visualizations and to log our metrics. This assignment borrows heavily from the [previous version](https://bitbucket.org/cmu16824_spring2020/2020_hw2_release/src/master/), but is now upgraded to Python 3, and does not depend upon the now deprecated Faster-RCNN repository.

## Readings

We will be implementing slightly simplified versions of the following approaches in this assignment:

1. Oquab, Maxime, et al. "*Is object localization for free?-weakly-supervised learning with convolutional neural networks.*" Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015. [Link](https://www.di.ens.fr/~josef/publications/Oquab15.pdf)
2. Bilen, Hakan, and Andrea Vedaldi. "*Weakly supervised deep detection networks*." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. [Link](https://www.robots.ox.ac.uk/~vgg/publications/2016/Bilen16/bilen16.pdf)


## Environment Setup

If you are using AWS instance setup using the provided instructions, you should already have most of the requirements installed on your machine. In any case, you would need the following Python libraries installed:

1. PyTorch
2. Weights and Biases
3. SKLearn
4. Pillow (PIL)
5. And many tiny dependencies that come pre-installed with anaconda or can be installed using ``conda install`` or ``pip install``

### Activate conda pytorch environment.
You can create a conda environment using the environment file provided to you:
```bash
conda env create -f environment.yml
```
If this doesn't work, feel free to install the dependencies using conda/pip. For reference, we used ``pytorch=1.11.0``, ``scikit-learn=0.23.2``, ``wandb=0.12.11``, ``pillow=9.0.1`` and Python 3.7. You will be using this environment for future assignments as well.

### Data setup

We will train and test using the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) data. The Pascal VOC dataset comes with bounding box annotations, however, we will not use bounding box annotations in the weakly-supervised setting.

1. We first need to download the image dataset and annotations. Use the following commands to setup the data, and let's say it is stored at location `$DATA_DIR`.
```bash
$ # First, cd to a location where you want to store ~0.5GB of data.
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ tar xf VOCtrainval_06-Nov-2007.tar
$ # Also download the test data
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007/
$ export DATA_DIR=$(pwd)
```
2. In the main folder of the code provided in this repository, there is an empty directory with the name `data`.
	- In this folder, you need to create a link to `VOCdevkit` in this folder.
	- For Task 2 (WSDDN [2]), we require bounding box proposals from Selective Search, Edge Boxes or a similar method. We provide you with this data for the assignment. You need to put these proposals in the data folder too.

```bash
# You can run these commands to populate the data directory
$ # First, cd to the main code folder
$ # Then cd to the data folder
$ cd data/VOCdevkit/VOC2007/
$ # Download the selective search data from https://drive.google.com/drive/folders/1jRQOlAYKNFgS79Q5q9kfikyGE91LWv1I to this location
```

## Task 0: Visualization and Understanding the Data Structures

### Modifying the Dataloader
First, you will have to modify the VOCDataset class in `voc_dataset.py` to return bounding boxes, classes corresponding to the bounding boxes, as well as selective search region proposals. Check the `TODO` in `voc_dataset.py` and make changes wherever necessary.

Once this is done, you will use Wandb to visualize the bounding boxes. The file `task_0.ipynb` has detailed instructions for this task.

#### Q 0.1: What classes does the image at index 2020 contain (index 2020 is the 2021-th image due to 0-based numbering)?
#### Q 0.2 Use Wandb to visualize the ground-truth bounding box and the class for the image at index 2020.
#### Q 0.3 Use Wandb to visualize the top ten bounding box proposals for the image at index 2020.


## Task 1: Is Object Localization Free?
Now that we have the data loaders set up, we're ready to get into object detectors! Before diving into object detectors though, let's see if we can localize objects using image-level classification labels. We'll be implementing a simplified version of the approach in [1], so you should go through the paper before getting started.

As proposed in [1], we will be using a trained ImageNet classification model and examine the backbone features to see if it provides us cues for the approximate locations of objects. We won't be training a classification network from scratch to save the rainforest (and AWS credits) but you should take a look at the code [here](https://github.com/pytorch/examples/blob/master/imagenet/main.py). We will be following the same structure.

First, we need to define our model. The code for the model is in `AlexNet.py`. In the code, you need to fill in the parts that say "TODO" (read the questions before you start filling in code). We are going to call this ``LocalizerAlexNet``. We've written a skeleton structure in `AlexNet.py`. You can look at the AlexNet example of PyTorch for reference.

For simplicity, we won't be copying the pre-trained FC layers to our model. Instead, we'll initialize new convolution layers and train them. This is quite different to [1], where the pre-trained FC layers are treated as convolutions. In summary, we want the model to look like this:
```text
LocalizerAlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
  )
  (classifier): Sequential(
    (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
    (3): ReLU(inplace)
    (4): Conv2d(256, 20, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

#### Q 1.1 Fill in each of the TODO parts in `AlexNet.py`. Next, fill the TODO parts in `task_1.py`except the functions ``metric1``, ``metric2`` and ``LocalizerAlexNetRobust``. You may need to refer to [1] for their choice of loss function and optimizer. As you may observe, the output of the above model has some spatial resolution. Make sure you read paper [1] and understand how to go from the output to an image-level prediction (max-pool). (Hint: This part will be implemented in ``train()`` and ``validate()``. For each of the TODO, describe the functionality of that part using appropriate comments.

#### Q 1.2 What is the shape (NxCxHxW) of the output of the model?

#### Plotting using Weights and Biases
Logging to [Weights and Biases](https://docs.wandb.ai/quickstart), also known as `wandb` is quite easy and super useful. You can use this to keep track of experiment hyperparameters and metrics such as loss/accuracy.
```python
import wandb
wandb.init(project="vlr-hw1")
# logging the loss
wandb.log({'epoch': epoch, 'loss': loss})
```
You can also use it to save models, perform hyperparameter tuning, share your results with others, collate different runs together and other cool stuff.

When you're logging to WandB, make sure you use good tag names. For example, for all training plots you can use ``train/loss``, ``train/metric1``, etc and for validation ``validation/metric1``, etc.

In this task, we will be logging losses, metrics, and images. Ensure that you're familiar with how to do these.

#### Q 1.3 Initialize the model from ImageNet (till the conv5 layer). Initialize the rest of layers with Xavier initialization and train the model using batchsize=32, learning rate=0.01, epochs=2 (Yes, only 2 epochs for now).(Hint: also try lr=0.1 - best value varies with implementation of loss)
- Use wandb to plot the training loss curve at every iteration.
- We also want to visualize where the network is "looking" for objects during classification. We can use the model's outputs for this. For example, to see where the class 0 is being localized, we can access the channel corresponding to class 0 in the model output. Use wandb to plot images and the rescaled heatmaps (to image resolution) for any GT class for 2 images at epoch 0 and epoch 1. The images and corresponding GT label should be the same across epochs so you can monitor how the network is learning to localize objects. (Hint: a heatmap has values between 0 and 1 while the model output does not!)
- Recommended training loss at the end of training: ~(0.15, 0.20)

#### Q 1.4 In the first few iterations, you should observe a steep drop in the loss value. Why does this happen? (Hint: Think about the labels associated with each image).

#### Q 1.5 We will log two metrics during training to see if our model is improving progressively with iterations. The first metric is mAP, a standard metric for multi-label classification. Write the code for this metric in the TODO block for ``metric1`` (make sure you handle all the boundary cases). However, ``metric1`` is to some extent not robust to the issue we identified in Q1.4. The second metric, Recall, is more tuned to this dataset. Even though there is a steep drop in loss in the first few iterations ``metric2`` should remain almost constant. Implement it in the TODO block for ``metric2``. (Make any assumptions needed - like thresholds). Feel free to use libraries like ``sklearn``.

### We're ready to train now!

#### Q 1.6 Initialize the model from ImageNet (till the conv5 layer), initialize the rest of layers with Xavier initialization and train the model using batchsize=32, learning rate=0.01, epochs=30. Evaluate every 2 epochs. (Hint: also try lr=0.1 - best value varies with implementation of loss) \[Expected training time: 45mins-75mins].
- IMPORTANT: FOR ALL EXPERIMENTS FROM HERE - ENSURE THAT THE SAME IMAGES ARE PLOTTED ACROSS EXPERIMENTS BY KEEPING THE SAMPLED BATCHES IN THE SAME ORDER. THIS CAN BE DONE BY FIXING THE RANDOM SEEDS BEFORE CREATING DATALOADERS.
- Use wandb to plot the training loss curve, training ``metric1``, training ``metric2`` at every iteration.
- Use wandb to plot the mean validation ``metric1`` and mean validation ``metric2`` every 2 epochs.
- Use wandb to plot images and the rescaled heatmaps for one of the GT classes for 2 images every 15 epochs (i.e., at the end of the 1st, 15th, and 30th epoch)
- At the end of training, use wandb to plot 3 randomly chosen images and corresponding heatmaps (similar to above) from the validation set.
- In your report, mention the training loss, training and validation ``metric1`` and ``metric2`` achieved at the end of training.
- Expected training metrics: ~(0.8, 0.9).
- Expected validation metrics: For metric1 ~(0.5, 0.6),
                               For metric2 ~(0.4, 0.5)


#### Q 1.7 In the heatmap visualizations you observe that there are usually peaks on salient features of the objects but not on the entire objects. How can you fix this in the architecture of the model? (Hint: during training the max-pool operation picks the most salient location). Implement this new model in ``LocalizerAlexNetRobust`` and also implement the corresponding ``localizer_alexnet_robust()``. Train the model using batchsize=32, learning rate=0.01, epochs=45. Evaluate every 2 epochs.(Hint: also try lr=0.1 - best value varies with implementation of loss)
- Hint:
    - You do not have to change the backbone AlexNet for implementing this. Think about how the network may try to use certain salient parts of the object more and what maybe a quick and easy way to prevent it.
- For this question only visualize images and heatmaps using wandb every 15 epochs as before (ensure that the same images are plotted).
- You don't have to plot the rest of the quantities that you did for previous questions (if you haven't put flags to turn off logging the other quantities, it's okay to log them too - just don't add them to the report).
- At the end of training, use wandb to plot 3 randomly chosen images (same images as Q1.6) and corresponding heatmaps from the validation set.
- Report the training loss, training and validation ``metric1`` and ``metric2`` achieved at the end of training.


## Task 2: Weakly Supervised Deep Detection Networks

First, make sure you understand the WSDDN model.

Now that we've implemented object detection, let us predict bounding boxes around the objects! Remember we visualized bounding boxes from the selective search data in Q0? Selective search is a region proposal algorithm that takes an image as the input and outputs bounding boxes corresponding to all patches in an image that are most likely to be objects. The ``BoxScores`` data point gives high scores for the bounding boxes with best overlap.

In WSDDN, the images and region proposals are taken as input into the ``spatial pyramid pooling`` (SPP) layer, both classification and detection is done on these regions, and class scores and region scores are computed & combined to predict the best possible bounding box. Understand this portion from the paper clearly.

The main script for training is ``task_2.py``.  Read all the comments to understand what each part does. There are a few major components that you need to work on:

- The network architecture and functionality ``WSDDN``
- Writing the traning loop and including visualizations for metrics
- The function `test_net()` in `task_2.py`, which will log metrics on the test set

Tip for Task 2: conda install tmux, and train in a tmux session. That way you can detach, close your laptop (don't stop your ec2 instance!), and go enjoy a cookie while you wait.

#### Q2.1 In ``wsddn.py``, you need to complete the  ``__init__, forward`` and `` build_loss`` functions.
The `__init__()` function will be used to define the model. You can define 3 parts for the model
1. The feature extractor
2. A ROI-pool layer (use torchvision.ops)
3. A classifier layer, as defined in the WSDDN paper.

The `forward()` function will essentially do the following:
1. Extract features from a given image (notice that batch size is 1 here). Feel free to use AlexNet code as backbone from the first part of the assignment.
2. Use the regions proposed from Selective Search, perform ROI Pooling. There are 2 caveats here - ensure that the proposals are now absolute values of pixels in the input image, and ensure that the scale factor passed into the ROI Pooling layer works correctly for the given image and features [ref](https://discuss.pytorch.org/t/spatial-scale-in-torchvision-ops-roi-pool/59270).
    - Note that for the scale factor in ROI Pooling closely depends on the coordinate values in your ROIs (i.e. whether these values are scaled or not). ROI output size should be a small value. Try out different values and see which works out best. Make sure you understand the ROI pooling API when using this function.
4. For each image, ROI Pooling gives us a feature map for the proposed regions. Pass these features into the classifier subnetwork. Here, you can think of batch size being the number of region proposals for each image.
5. Combine the classifier outputs (for boxes and classes), which will give you a tensor of shape (N_boxes x 20). Return this.

The `build_loss()` function now computes classification loss, which can be accessed in the training loop.


#### Q2.2 In ``task_2.py`` you will first need to write the training loop.
This involves creating the dataset, calling the dataloaders, creating the optimizer, etc. and then finally starting the training loop with the forward and backward passes. Some of this functionality has already been implemented for you. Ideally, use the hyperparameters given in the code. You don't need to implement the visualizations yet.
Use `top_n=300`, but feel free to increase it as well.

#### Q2.3 In ``task_2.py``, you now need to write a function to test your model, and visualize its predictions.
1. Write a test loop similar to the training loop, and calculate mAP as well as class-wise AP's.

To calculate the AP for each class correctly read this blog post:
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

For mAP, the computation is slightly different for Q2. The overall algorithm will then be something like:

1. Take all the model bbox predictions for your validation set and filter out those with low scores.
2. Now we want to find the precision/recall values (for each class separately) and for this we require TP and FP (Check the blog post).
3. To do this, iterate over your bbox predictions for the entire dataset
(Note: this is for a given class. You should iterate in sorted order based on predicted scores for entire validation set)
  - If there is no gt_bbox for this image mark your prediction as a "false positive (fp)"
  - But if there is some gt_bbox, find the iou(gt_bbox, pred_bbox), and if this iou value is greater than some threshold then mark it as a "tp" else mark it as a "fp"(again).
  - If you did find some gt_bbox that matched your prediction mark it as used since you cannot use it again. Now calculate precision and recall, precision should be tp/tp+fp and recall should be tp/allgtbboxes

At this point, we have our model giving us (N_boxes x 20) scores. We can interpret this as follows - for each of the 20 classes, there are `N` boxes, which have confidence scores for that particular class. Now, we need to perform Non-Max Suppression for the bbox scores corresponding to each class.
- In `utils.py`, write the NMS function. NMS depends on the calculation of Intersection Over Union (IoU), which you can either implement as a separate function, or vectorize within the NMS function itself. Use an IoU threshold of 0.3.
- Use NMS with a confidence threshold of 0.05 (basically consider only confidence above this value) to remove unimportant bounding boxes for each class.
- In the test code, iterate over indices for each class. For each class, visualize the NMSed bounding boxes with the class names and the confidence scores. You can use wandb for this, but if using ImageDraw or something else is more convenient, feel free to use that instead.


#### Q2.4 In ``task_2.py``, there are places for you perform visualization (search for TODO). You need to perform the appropriate visualizations mentioned here:
- Plot the average loss every 500 iterations (feel free to use the AverageMeter class from `task_1.py`) using wandb.
- Use wandb to plot mAP on the *test* set every epoch.
- Plot the class-wise APs at every epoch for at least 5 classes.
- Plot bounding boxes on 10 random images at the end of the first epoch, and at the end of the last epoch. (You can visualize for more images, and choose whichever ones you feel represent the learning of the network the best. It's also interesting to see the kind of mistakes the network makes as it is learning, and also after it has learned a little bit!)

#### Q2.5 Train the model using the hyperparameters provided for 5-6 epochs.
The expected values for the metrics at the end of training are:
- Train Loss: ~1.0
- Test  mAP : ~0.13

Some caveats for Train loss and Test mAP:
- If your loss does not go down or is too unstable, try lowering the learning rate.
- In case you have tried a lot and still cannot get a loss around ~1.0 then add *one plot for all of your valid tries* and add it to the report. Also, add 2-3 lines on what you believe is the reason for the observed behavior.
- Test AP (for detection) can show variance across different classes hence look at the mean value (mAP).

Include all the code and images/logs after training. Include appropriate comments on how you calculated class-wise AP and mAP.
Report the final class-wise AP on the test set and the mAP.


## Task 3: Visualizing Class Activation Maps (Extra Credit)

Another popular way to visualize how your network is learning is Class activation maps (as explained in the lecture)! In this task, we want you to experiment with state-of-the-art CAM methods (feel free to use existing implementations such as https://github.com/frgfm/torch-cam). Using a network you trained (in either task 1 or task 2), apply one (or more) CAM methods to visualize how well your network is doing. In the report, explain the implementation of the method you chose in 2-3 sentences. Additionally, add 2-3 visualizations in the report and try comparing and contrasting with the inferences you made using heatmaps.

 
# Submission Checklist 

In all the following tasks, coding and analysis, please write a short summary of what you tried, what worked (or didn't), and what you learned, in the report. Write the code into the files as specified. Submit a zip file (`ANDREWID.zip`) with all the code files, and a single `REPORT.pdf`, which should have commands that TAs can run to re-produce your results/visualizations etc. Also mention any collaborators or other sources used for different parts of the assignment.

## Report

### Task 0
- [ ] Answer Q0.1
- [ ] wandb screenshot for Q0.2
- [ ] wandb screenshot for Q0.3
### Task 1
- [ ] Q1.1 describe functionality of the completed TODO blocks with comments
- [ ] Answer Q1.2
- [ ] Q1.3
  - [ ] Add screenshot of training loss
  - [ ] Screenshot of wandb showing images and heat maps for the first logged epoch
  - [ ] Screenshot of wandb showing images and heat maps for the second logged epoch
- [ ] Answer Q1.4
- [ ] Answer Q1.5 and mention the assumptions
- [ ] Q1.6
	- [ ] Add screenshot of metric1, metric2 on the training set
	- [ ] Add screenshot of metric1, metric2 on the validation set
	- [ ] Screenshot of wandb showing images and heat maps for the first logged epoch \*show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the 15th logged epoch \*show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the last logged epoch \*show image and heatmap side-by-side\*.
	- [ ] wandb screenshot for 3 randomly chosen validation images and heat maps \*show image and heatmap side-by-side\*.
	- [ ] Report training loss, validation metric1, validation metric2 at the end of training

- [ ] Q1.7
	- [ ] Screenshot of wandb showing images and heat maps for the first logged epoch \*show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the 15th logged epoch \*show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the 30th logged epoch \*show image and heatmap side-by-side\*.
	- [ ] Screenshot of wandb showing images and heat maps for the last logged epoch \*show image and heatmap side-by-side\*.
	- [ ] wandb screenshot for 3 randomly chosen validation images (but same images as Q1.6) and heat maps \*show image and heatmap side-by-side\*.
	- [ ] Report training loss, validation metric1, validation metric2 at the end of training

### Task 2
- [ ] Q2.3 detailed code comments on how classwise AP and mAP are calculated
- [ ] Q2.4 wandb downloaded image of training loss vs iterations
- [ ] Q2.4 wandb downloaded image of test mAP vs iterations plot
- [ ] Q2.4 screenshot for class-wise APs vs iterations for 5 classes
- [ ] Q2.4 screenshot of 10 images with predicted boxes for the first logged epoch
- [ ] Q2.4 screenshot of 10 images with predicted boxes for the last logged epoch (~5 epochs)
- [ ] Q2.4 report final classwise APs on the test set and mAP on the test set

### Task 3
- [ ] Description of CAM method, 
- [ ] 2-3 images to compare similarities/differences with previous tasks

## Other Data
- [ ] code folder
