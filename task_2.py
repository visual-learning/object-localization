from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
import numpy as np
from datetime import datetime
import pickle as pkl

# imports
from wsddn import WSDDN
from voc_dataset import *
import wandb
from utils import nms, tensor_to_PIL
from PIL import Image, ImageDraw


# hyper-parameters
# ------------
start_step = 0
end_step = 20000
lr_decay_steps = {150000}
lr_decay = 1. / 10
rand_seed = 1024

lr = 0.0001
momentum = 0.9
weight_decay = 0.0005
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load datasets and create dataloaders

train_dataset = None
val_dataset = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,   # batchsize is one for this implementation
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    sampler=None,
    drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=True)


# Create network and initialize
net = WSDDN(classes=train_dataset.CLASS_NAMES)
print(net)

if os.path.exists('pretrained_alexnet.pkl'):
    pret_net = pkl.load(open('pretrained_alexnet.pkl', 'rb'))
else:
    pret_net = model_zoo.load_url(
        'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    pkl.dump(pret_net, open('pretrained_alexnet.pkl', 'wb'),
             pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

for name, param in pret_net.items():
    print(name)
    if name not in own_state:
        continue
    if isinstance(param, Parameter):
        param = param.data
    try:
        own_state[name].copy_(param)
        print('Copied {}'.format(name))
    except:
        print('Did not find {}'.format(name))
        continue


# Move model to GPU and set train mode
net.load_state_dict(own_state)
net.cuda()
net.train()

# TODO: Create optimizer for network parameters from conv2 onwards
# (do not optimize conv1)





output_dir = "./"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
disp_interval = 10
val_interval = 1000



def test_net(model, val_loader=None, thresh=0.05):
    """
    tests the networks and visualize the detections
    thresh is the confidence threshold
    """

    for iter, data in enumerate(val_loader):

        # one batch = data for one image
        image           = data['image']
        target          = data['label']
        wgt             = data['wgt']
        rois            = data['rois']
        gt_boxes        = data['gt_boxes']
        gt_class_list   = data['gt_classes']

        #TODO: perform forward pass, compute cls_probs


        # TODO: Iterate over each class (follow comments)
        for class_num in range(20):            
            # get valid rois and cls_scores based on thresh
            
            # use NMS to get boxes and scores
            

        #TODO: visualize bounding box predictions when required
        #TODO: Calculate mAP on test set




for iter, data in enumerate(train_loader):

    #TODO: get one batch and perform forward pass
    # one batch = data for one image
    image           = data['image']
    target          = data['label']
    wgt             = data['wgt']
    rois            = data['rois']
    gt_boxes        = data['gt_boxes']
    gt_class_list   = data['gt_classes']
    

    #TODO: perform forward pass - take care that proposal values should be in pixels for the fwd pass
    # also convert inputs to cuda if training on GPU



    

    # backward pass and update
    loss = net.loss    
    train_loss += loss.item()
    step_cnt += 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #TODO: evaluate the model every N iterations (N defined in handout)
    
    if iter%val_interval == 0 and iter != 0:
        net.eval()
        ap = test_net(net, val_loader)
        print("AP ", ap)
        net.train()


    #TODO: Perform all visualizations here
    #The intervals for different things are defined in the handout
