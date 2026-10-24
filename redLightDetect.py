'''
This is a project designed to make a redlight detector in two parts.
1. an openCV part, where it detects and highlights red
2. a neural net which takes the highlighted image for help in determining what to train/detect.
'''

import os # for checking if a saved net exists
import torch.nn as nn
import torch.nn.functional as F # for stateless operations where you can model weights yourself
import torch
import torchvision
from torchvision.transforms import v2
import torch.optim as optim

import numpy as np
import cv2 as cv # OpenCV (Computer Vision)
from PIL import Image # opencv images need to be turned into PIL for torchvision's transforms
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

#===========================================================#

# opencv part. its in functions now so the net can use it too, not just the final image.

min_area = 100 # CHANGE ME. red blobs smaller than this many pixels get ignored

def find_red(img): # steps 2-4 above. gives back the cleaned up red mask (white = red)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) # scan image and convert BGR to hsv

    lower_red = np.array([0,100,100]) # in hsv values which are weird 
    higher_red = np.array([10,255,255])

    # red is at BOTH ends of the hue range (0-10 and 170-180), so we need a second range or we miss half the reds
    lower_red2 = np.array([170,100,100])
    higher_red2 = np.array([180,255,255])

    mask = cv.bitwise_or(cv.inRange(hsv, lower_red, higher_red), cv.inRange(hsv, lower_red2, higher_red2))
    # bitwise_or = a pixel counts if it's in either red range

    kernel = np.ones((3,3), dtype=np.uint8) # 3,3 = anything smaller than 3x3px is removed.
    # 8 bits, because 255 bgr.

    noise_removed = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations = 1) 
    # takes mask, applies erosion followed by dilation under the directions of kernel iteration number of times.

    return noise_removed


def highlight_red(img): # the "highlighted image" the net looks at. red stays bright, everything else gets dimmed
    noise_removed = find_red(img)
    dimmed = (img * 0.3).astype(np.uint8) # CHANGE ME. 0.3 = how dark the not-red stuff gets
    return np.where(noise_removed[:, :, None] > 0, img, dimmed)
    # [:, :, None] makes the mask 3 channels deep so it lines up with the bgr image


def draw_red_lights(img, noise_removed): # steps 5-6 above. draws on img, gives back img and if any red was found

    contours, hierachy = cv.findContours(noise_removed, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    # hierachy is never used, it just breaks if you don't mention it.
    # RETR_LIST = just means, get all contours without hierachy.
    # CHAIN_APPROX_NONE = "stores absolutely all the contour points."

    found = False

    for contour in contours: # circles around individual red points instead of one big circle around everything

        if cv.contourArea(contour) < min_area: # tiny red specks aren't lights
            continue

        found = True

        cv.drawContours(img, [contour], contourIdx = -1, color = (255,255,255), thickness = 3)
        # -1 tells which contour to draw, -1 means all contours (just the one in the list here)

        (center_x, center_y), radius = cv.minEnclosingCircle(contour) # smallest circle that fits the whole contour
        center = (int(center_x), int(center_y))
        radius = math.ceil(radius)

        cv.circle(img, center, radius=(radius + 10), color=(0, 255, 0), thickness=2) # radius + 10 so it isn't super tight
        bottom_left = (max(center[0] - radius, 0), max(center[1] - radius - 20, 20)) # max() so the text doesn't go off the image
        cv.putText(img, "Red Light", bottom_left, cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), thickness = 2) # 0.75 = font size

    return img, found

#===========================================================#

'''
Neural net below:

Load and normalize the data (decide which traffic data source) training and test datasets using torchvision

Define a Convolutional Neural Network (CNN)

Define a loss function

Train the network on the training data

Test the network on the test data
'''

image_size = 64 # CHANGE ME. every image gets squished to this many px wide and tall, net needs them all the same size

def highlight_pil(pil_img): # lets the transform below use the opencv part. PIL is rgb, opencv is bgr
    img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    highlighted = highlight_red(img)
    return cv.cvtColor(highlighted, cv.COLOR_BGR2RGB)

transform = v2.Compose([ # turns image into something the neural net understands.
    v2.Resize((image_size, image_size)), # this is what makes them all one size
    v2.Lambda(highlight_pil), # opencv dims everything that isn't red, so the net gets the highlighted image
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True), # scale=True turns the 0-255 pixel values into 0-1 (it doesn't change image size)
    v2.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # "data normalization," turns your rgb image data into something the neural net can understand
    # i.e., decimals. std: output[channel] = (input[channel] - mean[channel]) / std[channel]
    # we can try overweighting red here, but that might lead to issues later down the line,
    # because you're overexposing the red colors to the model. for std
    # small std values will result in the output (of that channel) being empahsized.
    # v2.Normalize((0,0.5,0.5),(0.3,0.5,0.5)) <- if you want to try overempahsis to model.
])

batch_size = 10 # CHANGE ME number of images the ai is using for one iteration

# folders need to look like this, ImageFolder uses the folder names as the labels:
# data/train/red_light/*.jpg    data/train/not_red_light/*.jpg
# data/test/red_light/*.jpg     data/test/not_red_light/*.jpg
trainset = torchvision.datasets.ImageFolder(root = "data/train", transform = transform) # set of images for training

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True) # turns trainset into a PyTorch iterable, batch_size is changed elsewhere.
# deciding to shuffle for better training, might be too computation heavy, or just useless
# shuffle true, just out of fear it'll learn sequence patterns, instead of actual patterns.
# mem pin is unnecessary on laptop
# maybe use num_workers so its faster? another thing not for laptop?

# above has both test and answers
# below only has test given to AI, so we can test on real world

testset = torchvision.datasets.ImageFolder(root = "data/test", transform = transform) # set of images for testing

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle = False)
# shuffle off, because testdata isn't used for training.

classes = trainset.classes # ImageFolder sorts the folders alphabetically, so this is ('not_red_light', 'red_light'). 0 = not_red_light, 1 = red_light.
# (my og tuple had them the other way around, which would've swapped every label)

kernel = 3 # CHANGE ME. size of the filter window

class neuralNet(nn.Module): # the entire neural net, the brain.
    def __init__(self):
        super().__init__() # run initialization from parent (nn.Module), same as connect 4. breaks without it.

        self.conv1 = nn.Conv2d(3, 16, kernel_size = kernel, padding = kernel // 2) # switch output to 2 if looking for 
        # vertical lines (crosswalk) as well. only one for red right now. 
        # read to understand https://medium.com/@ml_dl_explained/understanding-2d-convolutions-in-pytorch-b35841149f5f
        # "filters" = feature detectors (look for bright red)
        # the reason we have so many "filters" (outputs) is because we need
        # to give it room to form an image of a red circle (bottom left curve + rest of curves...)
        # padding = kernel // 2 keeps the image the same size after the filter.
        self.pool = nn.MaxPool2d(2)
        # fairly straight-forward, conv2d outputs pos nums depending on how
        # many patterns/how strongly it thinks it sees patterns.
        # MaxPool2d just returns some of the highest positive nums (strongest patterns)
        # 2 is kernel (the viewing window/how big of an area you want to take max from), so 2x2.
        # (before it was 2*2 which is 4, so a 4x4 window)
        # stride is auto set to kernel size btw. so each pool halves the image.
        self.conv2 = nn.Conv2d(16, 32, kernel_size = kernel, padding = kernel // 2)
        # I don't really understand why the 2nd conv2d layer is needed.
        # more complexity I guess, the numbers I chose are utterly random.
        # output = CHANGE ME

        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), 120)
        # fc1 input = 32 filters * image width * image height after flatten. two pools = image is 4x smaller each side.
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 2) # 2 outputs, red_light and not_red_light

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
        # yep. crossentropyloss does the softmax itself, so we leave the raw scores.
        return x

net = neuralNet().to(device) # IMPORTANT. .to(device) sends the whole net to gpu/cpu

criterion = nn.CrossEntropyLoss() # https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5
# pk is the probablity of class k (correct), which is increased if correct
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9, weight_decay = 0.0001, nesterov = True) #.parameters() gives list of all weights and biases
# lr was 0.05 and weight_decay was 0.1, both way too big and the loss usually blows up or the weights get squashed to 0. CHANGE ME if training is too slow.
# also psuedo-random momentum (not with nesterov)
# and weight-decay CHANGE ME
# nesterov momentum is enabled, which gives momentum based on how the future 
# decision performs (seems like a no-brainer, research why not)
# no maximization, as we are looking for lowest error rate in chasing
# red traffic lights. maybe CHANGE ME if model doesn't work.

PATH = "./redlight.pt"

print("Do you want to train a new net, or use the saved one? (enter: train/use). Warning: training overwrites the saved net.")
choice = input()

if choice == "train":

    print_every = 20 # CHANGE ME. how many batches between loss prints. 2000 was too many for a small dataset, it never printed.

    for epoch in range(10): # loop over the dataset x amount of times. try ~50, reduced for performance CHANGE ME
    # training iterations are inside epochs.
    # training iteration: net + loss + backwardprop + optimizer
        running_loss = 0.0

        for i,data in enumerate (trainloader):

            inputs, labels = data # what you get from trainloader
            inputs, labels = inputs.to(device), labels.to(device) # the data has to be on the same device as the net

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

            running_loss += loss.item()
            if i % print_every == print_every - 1: # e.g. 19 % 20 == 19
                print(f"{epoch + 1}, {i + 1:5} loss: {running_loss / print_every:.5f}")
                #  1 + 1:5 tells code to     space the code five times
                # loss is divided by print_every because there are that
                # many running losses accumulated, 5f means 5 spaces and turn into float.
                running_loss = 0.0

    print("Training is done.")

    torch.save(net.state_dict(), PATH) # save the net in PATH
    # state_dict() = dict of the state of the pytorch tensor

assert os.path.exists(PATH), "No saved net found, enter 'train' first."

net.load_state_dict(torch.load(PATH, weights_only=True))
net.eval() # switches the net to test mode (turns off training-only stuff)

dataiter = iter(testloader)
images, labels = next(dataiter)

grid = torchvision.utils.make_grid(images) # sticks the whole batch into one picture
grid = grid / 2 + 0.5 # undoes the normalize (it did (x - 0.5) / 0.5)
grid = np.ascontiguousarray((grid.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
# tensor is (channels, h, w), opencv wants (h, w, channels). and * 255 to get back to normal pixel values.
grid = cv.cvtColor(grid, cv.COLOR_RGB2BGR) # torchvision is rgb, opencv is bgr
cv.imshow("Test images (what the net sees)", grid)
cv.waitKey(0)
cv.destroyAllWindows()

outputs = net(images.to(device))
_, predicted = torch.max(outputs, 1) # index of the highest score = the class the net picked

print("Real:     ", " ".join(classes[l] for l in labels))
print("Predicted:", " ".join(classes[p] for p in predicted))

correct = 0
total = 0

with torch.no_grad(): # no gradients, we aren't training so we don't need them
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item() # .cpu() so both are on the same device

print(f"Accuracy on the test images: {100 * correct / total:.1f}%")

#===========================================================#

# putting both parts together. opencv finds the red, then the net decides if its really a red light.

print("Please input the full file name of the image you want to scan (including extension):")

img = cv.imread(input()) # put full file name into terminal (including extension)

assert img is not None, "Please input the full file name of the image you want to scan (including extension):"

noise_removed = find_red(img)

image_contours, red_found = draw_red_lights(img.copy(), noise_removed)
# .copy() so the og image doesn't get drawn on, the net needs it clean

pil_img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # opencv is bgr, PIL wants rgb
net_input = transform(pil_img).unsqueeze(0).to(device) # unsqueeze adds the batch part, the net expects a batch of images

with torch.no_grad():
    outputs = net(net_input)
    _, predicted = torch.max(outputs, 1)

if red_found and classes[predicted.item()] == "red_light": # both have to agree
    print("red light detected")
    cv.imshow("Display", image_contours)
else:
    print("no red light detected")
    cv.imshow("Display", img)

cv.waitKey(15000)
cv.destroyAllWindows()
