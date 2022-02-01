import os
import cv2
import numpy as np
import pickle
from imutils import paths
from tensorflow.keras import models

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras  import Input

from keras.models import Model
from keras.applications import ResNet50
from keras.layers.pooling import AveragePooling2D

from keras.layers.core import Flatten
from keras.layers.core import Dense

from keras.layers.core import Dropout




datapath = r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\data"
outputmodel = r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\video_classification_model\videoclassificationmodel"
outputlabelbinarizer = r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\model\videoclassificationbinarizer"

Sports_Labels = set(['badminton', 'chess', 'cricket', 'football', 'swimming', 'weight_lifting'])
print("Loading ...")
PathToImages = list(paths.list_images(datapath))
data = []
labels = []

for images in PathToImages:
    label= images.split(os.path.sep)[-2]
    if label not in Sports_Labels:
        continue
    image = cv2.imread(images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(244,244))
    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(X_train,X_test,Y_train,Y_test) = train_test_split(data,labels,test_size = 25,stratify=labels,random_state = 42)

traininAugmentation = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest"
)
validationAugmention = ImageDataGenerator()
mean = np.array([123.68,116.779,103.939],dtype="float32")
traininAugmentation.mean = mean
validationAugmention.mean = mean



baseModel = ResNet50(weights="imagenet",include_top=False,input_tensor=Input(shape=(244,244,3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_),activation="softmax")(headModel)
model = Model(inputs=baseModel.input,outputs=headModel)

for baseModelLayers in baseModel.layers:
	baseModelLayers.trainable = False

from keras.optimizers import SGD
epochs = 35
opt = SGD(lr=0.0001,momentum=0.9,decay=1e-4/epochs)

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

History = model.fit_generator(
	traininAugmentation.flow(X_train,Y_train,batch_size=32),
	steps_per_epoch= len(X_train) // 32,
	validation_data=validationAugmention.flow(X_test,Y_test),
	validation_steps = len(X_test)//32,
	epochs=epochs
)

model.save(outputmodel)
lbinarizer = open(r"C:\Users\S0m3_7h1ng\Documents\Summer_Intern\model\videoclassificationbinarizer.pickle","wb")
lbinarizer.write(pickle.dumps(lb))
lbinarizer.close()