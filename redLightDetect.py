'''
This is a project designed to make a redlight detector in two parts.
1. an openCV part, where it detects and highlights red
2. a neural net which takes the highlighted image for help in determining what to train/detect.
'''

import torch.nn as nn
import torch.nn.functional as F # for stateless operations where you can model weights yourself
import torch
import torchvision
from torchvision.transforms import v2
import torch.optim as optim

import numpy as np
import cv2 as cv # OpenCV (Computer Vision)
import glob # for automated file detect. need to set up when net's done for automatic forwarding of images.
import math

''' 
how to find red light from image.

1. get image
2. get colors of image (through cv.inRange mask)
3. cv.imshow mask
4. filter mask, binary (ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY))
5. get contours of colored shape
6. draw rectangle around position?
7. return "red light detected," then add neural detection
'''

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" # will be gpu on pc
print(f"Using {device} device")

print("Please input the full file name of the image you want to scan (including extension):")

img = cv.imread("../redlight3.jpg") # put full file name into terminal (including extension)

assert img is not None, "Please input the full file name of the image you want to scan (including extension):"


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # scan image and convert BGR to hsv

lower_red = np.array([0,100,100]) # in hsv values which are weird 
higher_red = np.array([10,255,255])

mask = cv.inRange(hsv, lower_red, higher_red)

kernel = np.ones((3,3), dtype=np.uint8) # 3,3 = anything smaller than 3x3px is removed.
# 8 bits, because 255 bgr.

noise_removed = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1) 
# takes mask, applies erosion followed by dilation under the directions of kernel iteration number of times.


contours, hierachy = cv.findContours(noise_removed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# hierachy is never used, it just breaks if you don't mention it.
# RETR_LIST = just means, get all contours without hierachy.
# CHAIN_APPROX_NONE = "stores absolutely all the contour points."

image_contours = cv.drawContours(img, contours, contourIdx = -1, color = (255,255,255), thickness = 3)
# -1 tells which contour to draw, -1 means all contours


'''
find white pixels, group them and find their center, then draw
rectangle around said center. 
'''

y, x = np.where(np.all(image_contours == [255,255,255], axis = -1)) # you just need two input values for np.all()? axis = -1 means nothing in this context
# find all white from noise_removed, give index nums of positions.

# avg y,x? max and min y and x for radius. max and min x is diameter?

if len(x) > 0: # if white found, draw a circle around it and put "red light" near it
    diameter = max(x) - min(x)
    radius = math.ceil(diameter / 2)
    center_x = int(np.mean(x))
    center_y = int(np.mean(y))
    cv.circle(image_contours, (center_x, center_y), radius=(radius + 10), color=(0, 255, 0), thickness=2) # radius + 10 so it isn't super tight
    bottom_left = (center_x - 150), (center_y - 50)
    cv.putText(image_contours, "Red Light", bottom_left, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), thickness = 2) # 0.75 = font size
# maybe make so that it draws circles around individual red points?

cv.imshow("Display", image_contours)


cv.waitKey(15000)

#===========================================================#

'''
Neural net below:

Load and normalize the data (decide which traffic data source) training and test datasets using torchvision

Define a Convolutional Neural Network (CNN)

Define a loss function

Train the network on the training data

Test the network on the test data
'''

transform = v2.Compose([ # turns image into something the neural net understands.
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), # scale is the option ot scale the image inputs to uniform size
    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # "data normalization," turns your rgb image data into something the neural net can understand
    # i.e., decimals. std: output[channel] = (input[channel] - mean[channel]) / std[channel]
    # we can try overweighting red here, but that might lead to issues later down the line,
    # because you're overexposing the red colors to the model. for std
    # small std values will result in the output (of that channel) being empahsized.
    # v2.Normalize((0,0.5,0.5),(0.3,0.5,0.5)) <- if you want to try overempahsis to model.
])

batch_size = 10 # CHANGE ME number of images the ai is using for one iteration

trainset = torchvision.datasets.PUTIMAGEFOLDERHERE # set of images for training

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True) # turns trainset into a PyTorch iterable, batch_size is changed elsewhere.
# deciding to shuffle for better training, might be too computation heavy, or just useless
# shuffle true, just out of fear it'll learn sequence patterns, instead of actual patterns.
# mem pin is unnecessary on laptop
# maybe use num_workers so its faster? another thing not for laptop?

# above has both test and answers
# below only has test given to AI, so we can test on real world

testset = torch.vision.datasets.PUTIMAGEFOLDERHERE # set of images for testing 

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
# shuffle off, because testdata isn't used for training.

classes = ('red_light','not_red_light')

class neuralNet(nn.Module): # the entire neural net, the brain.
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size = kernel, padding = 1, device = device) # switch output to 2 if looking for 
        # vertical lines (crosswalk) as well. only one for red right now. 
        # read to understand https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f
        # "filters" = feature detectors (look for bright red)
        # the reason we have so many "filters" (outputs) is because we need
        # to give it room to form an image of a red circle (bottom left curve + rest of curves...)
        self.pool = nn.MaxPool2d((2*2), padding = 1)
        # fairly straight-forward, conv2d outputs pos nums depending on how
        # many patterns/how strongly it thinks it sees patterns.
        # MaxPool2d just returns some of the highest positive nums (strongest patterns)
        # 2*2 is kernel (the viewing window/how big of an area you want to take max from)
        # stride is auto set to kernel size btw.
        self.conv2 = nn.Conv2d(16, 32, kernel_size = kernel, padding = 1, device = device)
        # I don't really understand why the 2nd conv2d layer is needed.
        # more complexity I guess, the numbers I chose are utterly random.
        # output = CHANGE ME

        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 16)

        def forward(self, x): # used for how the input data is passed through layers
            x = self.pool(F.relu(self.conv1(x)))
            # inside -> out. scan for features with conv1,
            # make negative nums 0 with relu.
            # select max values with pool
            x = self.pool(F.relu(self.conv2(x)))
            # x here is acting as a container for the diff values, 
            # each function alters the data, but doesn't overwrite it.
            x = torch.flatten(x, 1)
            # flatten into a list/1d tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x) # why do we not call relu again? ah, because we want negative nums if its a bad choice?
            return x

net = neuralNet() # IMPORTANT

criterion = nn.CrossEntropyLoss() # https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
# pk is the probablity of class k (correct), which is increased if correct
optimizer = optim.SGD(net.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0.1, nesterov = True) #.parameters() gives list of all weights and biases
# high learning rate for low sample size CHANGE ME. also psuedo-random momentum (not with nesterov)
# and weight-decay CHANGE ME
# nesterov momentum is enabled, which gives momentum based on how the future 
# decision performs (seems like a no-brainer, research why not)
# no maximization, as we are looking for lowest error rate in chasing
# red traffic lights. maybe CHANGE ME if model doesn't work.


for epoch in range(10): # loop over the dataset x amount of times. try ~50, reduced for performance CHANGE ME
# training iterations are inside epochs.
# training iteration: net + loss + backwardprop + optimizer
    running_loss = 0.0

    for i,data in enumerate (trainloader):

        inputs, labels = data # what you get from trainloader

        optimizer.zero_grad() # resets gradient

        outputs = net(inputs) # put inputs into net
        loss = criterion(outputs, labels) 
        # takes correct+incorrect labels from trainloader
        # and neural outputs 
        loss.backward()
        # backprop works under diff wording.
        # The Leaves: the inputs user makes (data/weights)
        # The Braches: new nums from multilying/changing your inputs
        # The Trunk: when .backward called, gradient signal down tree
        optimizer.step()
        # updates model parameters using gradient from backprop.
        # think of a parabola, x^2, where the x-axis is the weight
        # value and the y-axis is the error rate.
        # optimizing is trying to find what weights give x = 0

        running_lost += loss.item()
        if i % 2000 == 1999: # 1999 % 2000 == 1999
            print(f"{epoch + 1}, {i + 1:5} loss: {running_loss / 2000:.5f}")
            #  1 + 1:5 tells code to     space the code five times
            # loss is divided by 2000 because there are that
            # many running losses accumulated, 5f means 5 spaces and turn into float.
            running_loss = 0.0

print("Training is done.")

PATH = "./redlight.pt"
torch.save(net.state_dict(), PATH) # save the net in PATH
# state_dict() = dict of the state of the pytorch tensor

dataiter = iter(testloader)
images, labels = next(dataiter)

cv.imshow(torchvision.utils.make_grid(images))

net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)
