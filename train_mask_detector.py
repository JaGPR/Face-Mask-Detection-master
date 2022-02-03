# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4 # provide learning rate to be low
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\prabh\Desktop\Face-Mask-Detection-master\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")

data = []#list having all images
labels = []#list having the lables like with mask and witjout mask

for category in CATEGORIES:
	path = os.path.join(DIRECTORY, category) # joining the two directory
	for img in os.listdir(path):# lsit dir list down all the particular directory in that directory
		img_path = os.path.join(path, img)# joining path with mask with a particular image
		image = load_img(img_path, target_size=(224, 224)) # this function is being called from kreas . preposiing , taget size is the height and width of the image
		image = img_to_array(image)# this function is being called from keras.preproseeing.image module
		image = preprocess_input(image) # we are using mobile nets ,learn this
		data.append(image)
		labels.append(category)
# perform one-hot encoding on the labels
lb = LabelBinarizer()# to convert the catery text to the numbers we are using sklearn lablebinarizer method
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data, labels,# it is used to split data
	test_size=0.20, stratify=labels, random_state=42)# only 20% of the images are given to the testing set and rest will be used for training
# train test split just splits the arrays into random train ansd test subset
# here we are using mobile net which is faster but it is less accurate as compared to its other competitors
# construct the training image generator for data augmentation
#image Data generator helps to create more images from the given image
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off

baseModel = MobileNetV2(weights="imagenet", include_top=False,#False is udesd to make fully connected layer at the topor not  #base ="imagenet means some pre initialised data set are their and their weights are being used here
	input_tensor=Input(shape=(224, 224, 3)))# input of the image and 3 here is the challen in coloured image that is r,g,b

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output# passing mobile net base model output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)#dense layer with 128 neurons and activation layer is relu for goto activation use cases
headModel = Dropout(0.5)(headModel)# overfitting of model
headModel = Dense(2, activation="softmax")(headModel)#output is 2 layer and activation function of softmnax or we can use sigma , dealing with binary so softmax

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)# 2 paramenter , head model as output
#wea are freezaing the layers here for training

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False#wea are freezaing the layers here for training

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) # adam is an optimizer #optimizxer is adam which is like adam ie like a goto optimizer
model.compile(loss="binary_crossentropy", optimizer=opt,#caluculatin of loss and learinig rate
	metrics=["accuracy"])# calculating the acuracy matrix here

# train the head of the network
print("[INFO] training head...")
H = model.fit(# we are going for image data gererator as we are having less data set
	aug.flow(trainX, trainY, batch_size=BS),#the iimage data been trained in start is being given here to get more train data
	steps_per_epoch=len(trainX) // BS,#validation data set test x and test y
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)# proving epochs as 20

# make predictions on the testing set]]
#evalution of our data
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
#here we are finding the index of the lable for the corresponding data set
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")# here we are saving our data inthe hFIVE (Hierarchical Data Format)FORMAT
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")# here weare saving two file one is the dataset file ans the other is the matplot lib png file'''