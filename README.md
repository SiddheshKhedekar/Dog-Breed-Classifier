
# Artificial Intelligence Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.



---
### Why We're Here

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>
## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
- `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
- `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels
- `dog_names` - list of string-valued dog breed names for translating labels


```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    #Load text files with categories as subfolder names.
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
```

    Using TensorFlow backend.


    There are 133 total dog categories.
    There are 8351 total dog images.

    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.


### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.


```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))

print(human_files[3])
```

    There are 13233 total human images.
    lfw/Donald_Rumsfeld/Donald_Rumsfeld_0118.jpg


---
<a id='step1'></a>
## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.


```python
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])

if img is None:
    print("There is an issue with cv2.imread")

# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1



![png](output_5_1.png)


Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.  

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.


```python
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

### (IMPLEMENTATION) Assess the Human Face Detector

__Question 1:__ Use the code cell below to test the performance of the `face_detector` function.  
- What percentage of the first 100 images in `human_files` have a detected human face?  
- What percentage of the first 100 images in `dog_files` have a detected human face?

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

__Answer:__ The algorithm detected a face with **100%** accuracy in the human files data set.

However, it also detected human faces in **11%** of the dog files data set which are false positives.


```python
import time

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.

human_count = 0
dog_count = 0

start_time = time.time()

for i in range(100):
    ## Test for Human faces
    if face_detector(human_files_short[i]):
        human_count += 1
    ## Test it on dog faces next
    if face_detector(dog_files_short[i]):
        dog_count +=1
stop_time = time.time()

duration = stop_time - start_time

print("Percentage of Human Faces detected in the Human Files is {}%".format(human_count))

print("Percentage of Human Faces detected in the Dog Files is {}%".format(dog_count))

print("Time taken in seconds for both detection algorithms on 100 samples each is : {:4.2f}".format(duration))
```

    Percentage of Human Faces detected in the Human Files is 100%
    Percentage of Human Faces detected in the Dog Files is 11%
    Time taken in seconds for both detection algorithms on 100 samples each is : 14.67


__Question 2:__ This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

__Answer:__ No, this does not seem like a reasonable expectation. For the algorithm to be used as a face detector in a useful real world application, it needs to be able to detect faces in real world images and not just in some specific situations.
For an algorithm to be useful and widely used, it needs to be robust and able to produce the desired result in many different environments.

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.


```python
## (Optional) TODO: Report the performance of another  
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>
## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.  

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!


```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).


```python
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).


```python
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
```

### (IMPLEMENTATION) Assess the Dog Detector

__Question 3:__ Use the code cell below to test the performance of your `dog_detector` function.  
- What percentage of the images in `human_files_short` have a detected dog?  
- What percentage of the images in `dog_files_short` have a detected dog?

__Answer:__
- Percentage of Dogs detected in the Human Files is 0%.
- Percentage of Dogs detected in the Dog Files is 100%


```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

human_count2 = 0
dog_count2 = 0

start_time = time.time()

for i in range(100):
    ## Test for Human faces
    if dog_detector(human_files_short[i]):
        human_count2 += 1
    ## Test it on dog faces next
    if dog_detector(dog_files_short[i]):
        dog_count2 +=1
stop_time = time.time()

duration = stop_time - start_time
```


```python
print("Percentage of Dogs detected in the Human Files is {}%".format(human_count2))

print("Percentage of Dogs detected in the Dog Files is {}%".format(dog_count2))

print("Time taken in seconds for both detection algorithms on 100 samples each is : {:4.2f}".format(duration))
```

    Percentage of Dogs detected in the Human Files is 0%
    Percentage of Dogs detected in the Dog Files is 100%
    Time taken in seconds for both detection algorithms on 100 samples each is : 26.16


---
<a id='step3'></a>
## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that *even a human* would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.  

Brittany | Welsh Springer Spaniel
- | -
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).  

Curly-Coated Retriever | American Water Spaniel
- | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">


Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.  

Yellow Labrador | Chocolate Labrador | Black Labrador
- | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.  

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.


```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras - Converts to (224, 224) and converts into a numpy array using PIL.
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

    100%|██████████| 6680/6680 [00:40<00:00, 164.93it/s]
    100%|██████████| 835/835 [00:04<00:00, 182.49it/s]
    100%|██████████| 836/836 [00:04<00:00, 178.83it/s]


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:

        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)

__Question 4:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

__Answer:__ Experiments were performed with a small Inception Network, a bigger Inception based Network and the network architecture described above. Implementation of each one of these can be looked at in the below functions.

The results of these experiments are tabulated below -

|Model | Num of Params | Time(s) | Loss | Val Acc(%) | Test Acc(%) |
|------|---------------|---------|------|------------|-------------|
|Small Inception| 1,333,253| 169| 4.89| 1.12| 1.1962|
|Big Inception| 2,671,909| 268| 4.98| 1.18| 1.1962|
|Default Network| 19,189| 128| 4.9779| 1.08| 1.1962|

The inception based networks were chosen because experimentally they have shown very good results on the End-End Image Classification challenge on the Image Net dataset.

Since the difference between dog breeds is quie subtle ( I tried to classify some of the images and I did very badly myself!), there is a need for a deeper network and more feature extractor layers are required.
An Inception layer architecture is well suited for this application.

The reason it did not perform much better than the default network suggested is the depth of the network. I was unable to use a network with multiple layers because of the restriction imposed by my GPU ( 2GB memory only).


```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input, concatenate
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, Model

```


```python
# GoogleNet (inception layer) which provides > 1% Accuracy -- Call this small Inception Model

def small_inception(input_shape):

    #LeNet based inception layer model will be used.
    input_img = Input(shape = input_shape)

    # First Convolution Layer
    conv_1 = Conv2D(4, (5,5), strides =(1,1), padding='same', activation='relu')(input_img)

    # Max Pool Layer
    maxpool1 = MaxPooling2D((2,2))(conv_1)

    # Second Conv layer
    conv_2 = Conv2D(4, (3,3), strides=(2,2), padding='same', activation='relu')(maxpool1)

    # First inception Layer
    path1_1 = Conv2D(8, (1,1), padding='same', activation='relu')(conv_2)
    path1_1 = Conv2D(8, (3,3), padding='same', activation='relu')(path1_1)

    path1_2 = Conv2D(8, (1,1), padding='same', activation='relu')(conv_2)
    path1_2 = Conv2D(8, (5,5), padding='same', activation='relu')(path1_2)

    path1_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(conv_2)
    path1_3 = Conv2D(8, (1,1), padding='same', activation='relu')(path1_3)

    inception_out1 = concatenate([path1_1, path1_2, path1_3], axis = 3)

    maxpool2 = MaxPooling2D((2,2))(inception_out1)

    interim_1 = Conv2D(4, (1,1), padding='same', activation='relu')(maxpool2)

    fc1 = Flatten()(interim_1)

    out = Dense(133, activation='softmax')(fc1)

    model = Model(inputs = input_img, outputs = out)

    model.summary()

    return model

```


```python
# Bigger Version of the LeNet insipred Inception Layer based network

def big_inception(input_shape):

    #LeNet based inception layer model will be used.
    input_img = Input(shape = input_shape)

    # First Convolution Layer
    conv_1 = Conv2D(8, (5,5), strides =(2,2), padding='valid', activation='relu')(input_img)

    # Max Pool Layer
    maxpool1 = MaxPooling2D((3,3), strides=(1,1), padding='valid')(conv_1)

    # Second Conv layer
    conv_2 = Conv2D(8, (5,5), strides=(2,2), padding='valid', activation='relu')(maxpool1)

    # First inception Layer
    path1_1 = Conv2D(16, (1,1), padding='same', activation='relu')(conv_2)
    path1_1 = Conv2D(16, (3,3), padding='same', activation='relu')(path1_1)

    path1_2 = Conv2D(16, (1,1), padding='same', activation='relu')(conv_2)
    path1_2 = Conv2D(16, (5,5), padding='same', activation='relu')(path1_2)

    path1_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(conv_2)
    path1_3 = Conv2D(16, (1,1), padding='same', activation='relu')(path1_3)

    inception_out1 = concatenate([path1_1, path1_2, path1_3], axis = 3)

    maxpool2 = MaxPooling2D((3,3), strides=(1,1), padding='valid')(inception_out1)

    interim_1 = Conv2D(8, (1,1), padding='same', activation='relu')(maxpool2)

    fc1 = Flatten()(interim_1)

    out = Dense(133, activation='softmax')(fc1)

    model = Model(inputs = input_img, outputs = out)

    model.summary()

    return model
```


```python
def default_model(input_shape):

    input_img = Input(shape = input_shape)

    # First Conv 2D layer
    conv_1 = Conv2D(16, (2,2), padding='valid',activation='relu')(input_img)
    # Max Pooling Layer
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
    # Second Conv 2D layer
    conv_2 = Conv2D(32, (2,2),padding='valid',activation='relu')(maxpool1)
    # Max Pooling Layer
    maxpool2 = MaxPooling2D((2,2))(conv_2)
    # Third Conv 2D layer
    conv_3 = Conv2D(64,(2,2), padding='valid',activation='relu')(maxpool2)
    # Max Pooling Layer
    maxpool3 = MaxPooling2D((2,2))(conv_3)
    # Global Average Pooling Layer
    globalavgpool = GlobalAveragePooling2D()(maxpool3)
    # Fully Connected Layer
    out = Dense(133, activation='softmax')(globalavgpool)
    model = Model(inputs = input_img, outputs = out)

    model.summary()

    return model

```


```python
# Set the input_shape
input_shape = (224, 224, 3)

# Set the CNN model that you would like to test
# It can be 1. small_inception 2. big_inception 3. default_model

# Set the Model that you would like to test
test_model = "default_model"

if test_model == 'small_inception':
    model = small_inception(input_shape)
if test_model =='big_inception':
    model = big_inception(input_shape)
if test_model == 'default_model':
    model = default_model(input_shape)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 224, 224, 3)       0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 223, 223, 16)      208       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 111, 111, 16)      0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 110, 110, 32)      2080      
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 55, 55, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 54, 54, 64)        8256      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 27, 27, 64)        0         
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 64)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               8645      
    =================================================================
    Total params: 19,189
    Trainable params: 19,189
    Non-trainable params: 0
    _________________________________________________________________


### Compile the Model


```python
#model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.


```python
from keras.callbacks import ModelCheckpoint  

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 3

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=1, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/3
    6677/6680 [============================>.] - ETA: 0s - loss: 4.8904 - acc: 0.0099Epoch 00001: val_loss improved from inf to 4.88261, saving model to saved_models/weights.best.from_scratch.hdf5
    6680/6680 [==============================] - 119s 18ms/step - loss: 4.8905 - acc: 0.0099 - val_loss: 4.8826 - val_acc: 0.0108
    Epoch 2/3
    6678/6680 [============================>.] - ETA: 0s - loss: 4.9086 - acc: 0.0121Epoch 00002: val_loss did not improve
    6680/6680 [==============================] - 119s 18ms/step - loss: 4.9086 - acc: 0.0121 - val_loss: 4.9184 - val_acc: 0.0108
    Epoch 3/3
    6678/6680 [============================>.] - ETA: 0s - loss: 4.9254 - acc: 0.0163Epoch 00003: val_loss did not improve
    6680/6680 [==============================] - 107s 16ms/step - loss: 4.9252 - acc: 0.0163 - val_loss: 4.9353 - val_acc: 0.0228





    <keras.callbacks.History at 0x7fac58453198>



### Load the Model with the Best Validation Loss


```python
model.load_weights('saved_models/weights.best.from_scratch.hdf5')
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.


```python
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# Find out the Number of Samples
num_test_samples = len(dog_breed_predictions)

print("The Number of Test images are: {}".format(num_test_samples))

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    The Number of Test images are: 836
    Test accuracy: 1.1962%


---
<a id='step4'></a>
## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features


```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

print("Shape of train_VGG16: {}".format(train_VGG16.shape))
print("Shape of valid_VGG16: {}".format(valid_VGG16.shape))
print("Shape of test_VGG16: {}".format(test_VGG16.shape))
```

    Shape of train_VGG16: (6680, 7, 7, 512)
    Shape of valid_VGG16: (835, 7, 7, 512)
    Shape of test_VGG16: (836, 7, 7, 512)


### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.


```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_2 ( (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 133)               68229     
    =================================================================
    Total params: 68,229
    Trainable params: 68,229
    Non-trainable params: 0
    _________________________________________________________________


### Compile the Model


```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model


```python
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets,
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=8, callbacks=[checkpointer], verbose=1)
```

    Train on 6680 samples, validate on 835 samples
    Epoch 1/20
    6576/6680 [============================>.] - ETA: 0s - loss: 12.6119 - acc: 0.1201Epoch 00001: val_loss improved from inf to 11.54428, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 4s 616us/step - loss: 12.5937 - acc: 0.1211 - val_loss: 11.5443 - val_acc: 0.1880
    Epoch 2/20
    6672/6680 [============================>.] - ETA: 0s - loss: 10.9401 - acc: 0.2507Epoch 00002: val_loss improved from 11.54428 to 10.96976, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 508us/step - loss: 10.9391 - acc: 0.2509 - val_loss: 10.9698 - val_acc: 0.2491
    Epoch 3/20
    6592/6680 [============================>.] - ETA: 0s - loss: 10.6123 - acc: 0.2938Epoch 00003: val_loss improved from 10.96976 to 10.79131, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 409us/step - loss: 10.6097 - acc: 0.2943 - val_loss: 10.7913 - val_acc: 0.2743
    Epoch 4/20
    6552/6680 [============================>.] - ETA: 0s - loss: 10.2945 - acc: 0.3138Epoch 00004: val_loss improved from 10.79131 to 10.38075, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 363us/step - loss: 10.2754 - acc: 0.3150 - val_loss: 10.3808 - val_acc: 0.2790
    Epoch 5/20
    6600/6680 [============================>.] - ETA: 0s - loss: 9.6600 - acc: 0.3555Epoch 00005: val_loss improved from 10.38075 to 9.98215, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 408us/step - loss: 9.6677 - acc: 0.3543 - val_loss: 9.9822 - val_acc: 0.3066
    Epoch 6/20
    6640/6680 [============================>.] - ETA: 0s - loss: 9.3551 - acc: 0.3783Epoch 00006: val_loss improved from 9.98215 to 9.74973, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 380us/step - loss: 9.3524 - acc: 0.3786 - val_loss: 9.7497 - val_acc: 0.3293
    Epoch 7/20
    6576/6680 [============================>.] - ETA: 0s - loss: 9.1073 - acc: 0.3996Epoch 00007: val_loss improved from 9.74973 to 9.51607, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 394us/step - loss: 9.0966 - acc: 0.4006 - val_loss: 9.5161 - val_acc: 0.3509
    Epoch 8/20
    6520/6680 [============================>.] - ETA: 0s - loss: 8.8883 - acc: 0.4115Epoch 00008: val_loss improved from 9.51607 to 9.25848, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 383us/step - loss: 8.8602 - acc: 0.4132 - val_loss: 9.2585 - val_acc: 0.3533
    Epoch 9/20
    6560/6680 [============================>.] - ETA: 0s - loss: 8.5525 - acc: 0.4378Epoch 00009: val_loss improved from 9.25848 to 9.11947, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 406us/step - loss: 8.5437 - acc: 0.4382 - val_loss: 9.1195 - val_acc: 0.3641
    Epoch 10/20
    6592/6680 [============================>.] - ETA: 0s - loss: 8.2662 - acc: 0.4612Epoch 00010: val_loss improved from 9.11947 to 8.79744, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 387us/step - loss: 8.2705 - acc: 0.4611 - val_loss: 8.7974 - val_acc: 0.3916
    Epoch 11/20
    6640/6680 [============================>.] - ETA: 0s - loss: 8.0911 - acc: 0.4744Epoch 00011: val_loss improved from 8.79744 to 8.78383, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 397us/step - loss: 8.0993 - acc: 0.4738 - val_loss: 8.7838 - val_acc: 0.3892
    Epoch 12/20
    6544/6680 [============================>.] - ETA: 0s - loss: 7.8435 - acc: 0.4867Epoch 00012: val_loss improved from 8.78383 to 8.56716, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 388us/step - loss: 7.8639 - acc: 0.4856 - val_loss: 8.5672 - val_acc: 0.3976
    Epoch 13/20
    6640/6680 [============================>.] - ETA: 0s - loss: 7.6772 - acc: 0.5033Epoch 00013: val_loss improved from 8.56716 to 8.53469, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 387us/step - loss: 7.6821 - acc: 0.5031 - val_loss: 8.5347 - val_acc: 0.4084
    Epoch 14/20
    6600/6680 [============================>.] - ETA: 0s - loss: 7.6355 - acc: 0.5095Epoch 00014: val_loss improved from 8.53469 to 8.41057, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 4s 641us/step - loss: 7.6357 - acc: 0.5097 - val_loss: 8.4106 - val_acc: 0.4132
    Epoch 15/20
    6672/6680 [============================>.] - ETA: 0s - loss: 7.5830 - acc: 0.5138Epoch 00015: val_loss improved from 8.41057 to 8.27022, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 4s 632us/step - loss: 7.5788 - acc: 0.5141 - val_loss: 8.2702 - val_acc: 0.4216
    Epoch 16/20
    6624/6680 [============================>.] - ETA: 0s - loss: 7.3974 - acc: 0.5252Epoch 00016: val_loss improved from 8.27022 to 8.26473, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 4s 596us/step - loss: 7.3961 - acc: 0.5253 - val_loss: 8.2647 - val_acc: 0.4323
    Epoch 17/20
    6536/6680 [============================>.] - ETA: 0s - loss: 7.2970 - acc: 0.5311Epoch 00017: val_loss improved from 8.26473 to 8.06787, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 2s 370us/step - loss: 7.2826 - acc: 0.5320 - val_loss: 8.0679 - val_acc: 0.4395
    Epoch 18/20
    6584/6680 [============================>.] - ETA: 0s - loss: 7.2238 - acc: 0.5412Epoch 00018: val_loss did not improve
    6680/6680 [==============================] - 3s 389us/step - loss: 7.2160 - acc: 0.5415 - val_loss: 8.0862 - val_acc: 0.4347
    Epoch 19/20
    6600/6680 [============================>.] - ETA: 0s - loss: 7.1708 - acc: 0.5427Epoch 00019: val_loss did not improve
    6680/6680 [==============================] - 3s 384us/step - loss: 7.1740 - acc: 0.5424 - val_loss: 8.1751 - val_acc: 0.4251
    Epoch 20/20
    6616/6680 [============================>.] - ETA: 0s - loss: 6.9910 - acc: 0.5506Epoch 00020: val_loss improved from 8.06787 to 7.93338, saving model to saved_models/weights.best.VGG16.hdf5
    6680/6680 [==============================] - 3s 441us/step - loss: 6.9991 - acc: 0.5501 - val_loss: 7.9334 - val_acc: 0.4395





    <keras.callbacks.History at 0x7fac58394ac8>



### Load the Model with the Best Validation Loss


```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.


```python
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 45.5742%


### Predict Dog Breed with the Model


```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>
## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz

where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']


```python
### TODO: Obtain bottleneck features from another pre-trained CNN.

# Compare all four witht same network and see which one is better

# Choose 1 of the following four 'ResNet-50', 'Inception', 'Xception', 'VGG-19'

network = 'Xception'

if network =='ResNet-50':
    bottleneck_features_network = np.load('bottleneck_features/DogResnet50Data.npz')
elif network == 'Inception':
    bottleneck_features_network = np.load('bottleneck_features/DogInceptionV3Data.npz')
elif network =='Xception':
    bottleneck_features_network = np.load('bottleneck_features/DogXceptionData.npz')
elif network =='VGG-19':
    bottleneck_features_network = np.load('bottleneck_features/DogVGG19Data.npz')

train_network = bottleneck_features_network['train']
valid_network = bottleneck_features_network['valid']
test_network = bottleneck_features_network['test']

print("Shape of train_resnet: {}".format(train_network.shape))
print("Shape of valid_resnet: {}".format(valid_network.shape))
print("Shape of test_resnet: {}".format(test_network.shape))
```

    Shape of train_resnet: (6680, 7, 7, 2048)
    Shape of valid_resnet: (835, 7, 7, 2048)
    Shape of test_resnet: (836, 7, 7, 2048)


### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:

        <your model's name>.summary()

__Question 5:__ Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

__Answer:__ The details of the experiments performed and the final Network chosen are given below.


### Performance of different network using Transfer learning

#### Using very simple Final Layer

All the four different networks have been tested. In this case, additional layers just include a Global Average Pooling layer followed by a softmax activation as hown in the code below.

```python
Network_model = Sequential()
Network_model.add(GlobalAveragePooling2D(input_shape=train_network.shape[1:]))
Network_model.add(Dense(133, activation='softmax'))
```
The number of parameters varies depending on the final shape of the last layer of the network that has been explored. Since, these do not directly indicate the number of parameters in the trained network, this has not been included as a relevant parameter anymore.

The results are summarized below, the most important parameters in this case would be the different accuracies.

All the models have been trained for 20 epochs and the time mentioned is for training one epoch.

|Model      | Time (s) | Train Acc (%) | Val Acc (%) | Test Acc (%) |
|-----------|----------|---------------|-------------|--------------|
|VGG-19     | 4        | 46.8          | 37.25       | 39.59 |
|ResNet-50  | 7 | 98.59 | 76.29 | 78.46 |
|Inception | 7 | 98.95 | 84.91 | 76.07 |
|Xception | 8 | 99.7 | 84.83 | 84.09 |

#### Using deeper layer with 2 Conv layers on top of Xception network

As you can see from the results above, all networks except for the VGG-19 have a test accuracy of higher than 60% with the Xcpetion network performing the best at 84%.
More importantly though, there is a large difference between training and val/test accuracy for all the networks.

This can be attributed to either
- The size of the dataset is not large enough
- The avoidable bias is very large because the network is not deep enough.

To try to reduce the avoidable bias, an experiment with additional CNN layers was performed. The CNN had to be small so as to take full advantage of training using a GPU (2GB Memory).

The test accuracy was however only 65% accuracy, which was much lower than that achieved with Xception + Simple Fully Connected layers.

The best architecture based on transfer learning from an Xception is what is used for the algorithm further below.

### Why does the Xception Network do the best ?

The VGG-19 network consists of 19 traditional simple convolutional layers which are trained to learn image features. Hence, the result of training using this network results in a fairly good accuracy of around ~40%.

The ResNet, Inception and Xception architectures, however, use architectures (skip layers in the case of ResNet and Inception modules in the case of Inception and Xception) to enable training much deeper networks without over-fitting.

The Xception Network in particular uses an extreme Inception module architecure, which consists of 1x1 convolution filters, followed by multiple 3x3 or 5x5 filters, thus decoupling the mapping of cross-channels correlations and spatial correlations in the feature maps of convolutional neural networks [1]

![Xception Module Image taken from the paper](images/Xception_Module.png)

References
[1]	Xception: Deep Learning with Depthwise Separable Convolutions. arXiv:1610.02357


```python
### TODO: Define your architecture.

# Test by adding very simple layers

# Two different kinds of Architecture have been tried 1. simple and 2. cnn
architecture = 'simple'

if architecture =='simple':
    Network_model = Sequential()
    Network_model.add(GlobalAveragePooling2D(input_shape=train_network.shape[1:]))
    Network_model.add(Dense(133, activation='softmax'))

if network=='Xception' and architecture=='cnn':
    print("Using additional CNN layers along with transfer Learning from Xception Network...")

    Network_model = Sequential()

    Network_model.add(Conv2D(256,(1,1), padding='same',activation='relu',input_shape=train_network.shape[1:]))
    #Network_model.add(MaxPooling2D(pool_size=(2,2)))
    #Network_model.add(Dropout(0.25))

    Network_model.add(Conv2D(256,(3,3), padding='same',activation='relu'))
    Network_model.add(MaxPooling2D(pool_size=(2,2)))
    #Network_model.add(Dropout(0.25))

    Network_model.add(Flatten())
    Network_model.add(Dense(133,activation='softmax'))


Network_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    global_average_pooling2d_3 ( (None, 2048)              0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 133)               272517    
    =================================================================
    Total params: 272,517
    Trainable params: 272,517
    Non-trainable params: 0
    _________________________________________________________________


### (IMPLEMENTATION) Compile the Model


```python
### TODO: Compile the model.
Network_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.  

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.


```python
### TODO: Train the model.
from keras.callbacks import ModelCheckpoint
import os

model_filepath = 'saved_models/weights.best.' + network + '.hd5'

# Check if the file exists already. If it does, dont train again, move on to loading the best weights.

if os.path.isfile(model_filepath):
    print("The model has been trained and the weights already exist. Move on to loading..")
else:
    print("The weights will be saved to: {}".format(model_filepath))

    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=True)

    ## Point to remember --> train_targets, valid_targets, test_targets contains the one-hot encoded correct values.

    Network_model.fit(train_network, train_targets,
              validation_data=(valid_network, valid_targets),
              epochs=10, batch_size=1, callbacks=[checkpointer], verbose=1)
```

    The model has been trained and the weights already exist. Move on to loading..


### (IMPLEMENTATION) Load the Model with the Best Validation Loss


```python
### TODO: Load the model weights with the best validation loss.
Network_model.load_weights(model_filepath)

print("The best weights for the {} model have been loaded".format(network))
```

    The best weights for the Xception model have been loaded


### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.


```python
### TODO: Calculate classification accuracy on the test dataset.

predictions = [np.argmax(Network_model.predict(np.expand_dims(feat, axis=0))) for feat in test_network]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```

    Test accuracy: 84.0909%


### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.  

Similar to the analogous function in Step 5, your function should have three steps:
1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}

where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.


```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.
from extract_bottleneck_features import *

def network_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Network_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step6'></a>
## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
- if a __dog__ is detected in the image, return the predicted breed.
- if a __human__ is detected in the image, return the resembling dog breed.
- if __neither__ is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are __required__ to use your CNN from Step 5 to predict dog breed.  

Some sample output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_output.png)


### (IMPLEMENTATION) Write your Algorithm


```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.
def detect_breed(image_path):
    if face_detector(image_path):
        print("Hello, human!")
    elif dog_detector(image_path):
        print("Hello, dog!")
    else:
        print("Error: Neither a human face or a dog was detected.\n")
        return
    # Use same Image Pipeline as used earlier
    img = cv2.imread(image_path)
    # Convert from BGR to RGB
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Plot the
    plt.imshow(cv_rgb)
    plt.show()

    print("You look like a ...")
    print(network_predict_breed(image_path))
    print()
```

---
<a id='step7'></a>
## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that __you__ look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images!

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.  

__Question 6:__ Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

__Answer:__ The output is better than I expected. While I do not really have a good way of quantifying how well the classifier is working on the human images, the algorithm recognized all 3 stock photos ( all different sizes and poses,) of 3 different breeds of dogs and this I thought was pretty impressive was a network trained using transfer learning is under 3 minutes ( 7s for each epoch).

It does not mistake a cat for a dog in the experiments that I performed. An example if shown below.

Three things to improve the accuracy of the algorithm -
1.


```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.
# See how it does on humans
detect_breed("images/natalie_portman.jpg")
detect_breed("images/Mona_Lisa.jpg")
detect_breed("images/George-W-Bush.jpg")
detect_breed("images/Richard_Nixon.jpg")
# See how it does on Dogs
detect_breed("images/Newfoundland.jpeg")
detect_breed("images/Saint_Bernard.jpg")
```

    Hello, human!



![png](images/natalie_portman.jpg)


    You look like a ...
    Anatolian_shepherd_dog

    Hello, human!



![png](images/Mona_Lisa.jpg)


    You look like a ...
    Newfoundland

    Hello, human!



![png](images/George-W-Bush.jpg)


    You look like a ...
    Lowchen

    Hello, human!



![png](images/Richard_Nixon.jpg)


    You look like a ...
    Lowchen

    Hello, dog!



![png](images/Newfoundland.jpeg)


    You look like a ...
    Newfoundland

    Hello, dog!



![png](images/Saint_Bernard.jpg)


    You look like a ...
    Saint_bernard




```python
detect_breed("images/alaskan_malamute.jpg")
```

    Hello, dog!



![png](images/alaskan_malamute.jpg)


    You look like a ...
    Alaskan_malamute




```python
# Try this on a random Cat
print("The algorithm is being used on stock cat photo")

detect_breed("images/random_cat.jpg")
```

    The algorithm is being used on stock cat photo
    Error: Neither a human face or a dog was detected.




```python

```
