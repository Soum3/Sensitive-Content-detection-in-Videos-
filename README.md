CODE

#Cell 1: Importing necessary libraries

import cv2

Import os

import numpy as np

import tensorflow as tf

from tensorflow.keras.applications import

InceptionV3

from tensorflow.keras.models import Model from tensorflow.keras.layers import Input, LSTM, Bidirectional, BatchNormalization, Dropout, Dense

from sklearn.model selection import

train test _split

From tadm import todn from sklearn.metrics import

classification_report

#Cell 2: Loading the InceptionV3 model for feature extraction

def load pretrained model():

pretrained_model = InceptionV3()

# Extract features from the second-to-last layer of the InceptionV3 model

pretrained_model =

Model (inputs=pretrained model.inpu t

outputs pretrained_model.layers[- 2].output)

return pretrained model

pretrained model = load_pretrained_model()

pretrained model.summary()Cell 3: Function for extracting Features from frames

def extract frame features (frame, pretrained_model):

#Expand the dimensions of the Frame for model compatibility

img = np.expand_dims (frame, axis=0)

#Use the pre-trained feature extraction model to obtain the feature vector

feature vector

pretrained model.predict(img, verbose=0

#Return the extracted feature

vector

return feature_vector

Cell 4: Function for extracting

21/35

frames from videos

def

extract video frames (video_path,

sequence length=16,

image_width=299, image_height=299):

frames list =

#Open the video file for

reading

video reader = cv2.VideoCapture(video_path)

#Get the total number of frames in the video

video frames_count = int(video_reader.get(cv2.CAP PROP FRAME_COUNT))

#Calculate the number of Frames to skip in order to achieve the desired sequence length

skip frames_window= max(int(video_frames_count / sequence length), 1)#Loop through each frame in

the sequence

for frame_counter in range (sequence_length):

#Set the position of the video reader to the current frame

video_reader.set(cv2.CAP P

ROP POS FRAMES, frame_counter * skip_frames_window)

#Read the frame

success, frame =

video_reader.read()

#Break if unable to read

the frame

if not success:

break

#Convert the frame to RGB and

resize it

frame_rgb = cv2.cvtColor(frame cv2.COLOR BGR2RGB)

23/35

resized frame =

cv2.resize(frame_rgb, (image_height, image_width))

#Append the resized frame to the frames list

frames list.append(resized

frame)

#Release the video reader

video_reader.release()

#Return the list of frames

return frames listCell 5: Function for

ires from

videos

def extract_features_from_videos (video paths, total videos, pretrained model):

all video features = []

#Loop through each video

for pos in tqdm(range(total_videos)):

frames list = []

Extract frames from the current video

frames =

extract_video_frames(video_paths [pos])

#Extract features from each frame

for frame in frames:

Features extract frame_features(frame, pretrained_model)

frames list.append(features)

all_video_features.append(frames_list)

return np.array(all_video features)

#Cell 6: Loading features and preparing data for model training

25/35

Define violence and non-violence directories

violence_dir = '/kaggle/input/real- life-violence-situations-dataset/Real Life Violence Dataset/Violence' nonviolence dir /kaggle/input/real- life-violence-situations-dataset/Real Life Violence Dataset/Nonviolence'

#Create paths to individual videos

violence path = [os.path.join(violence dir, name) for name in os.listdir (violence dir)]

nonviolence_path =

[os.path.join(nonviolence_dir, name) for name in os.listdir(nonviolence_dir#Extract features from videos

violence features

extract features from videos (viole

nce_path[:500],

len(violence_path[:500]),

pretrained model)

non violence_features

extract features_from_videos (nonvi olence_path[:500], len(nonviolence_path[:500]), pretrained model)

#Save extracted features

np.save('/kaggle/working/violence Features.npy, violence_features) np.save('/kaggle/working/non_viole nce features.npy', non violence features)

Cell 7: Loading features and labels for model training

Load features and labels

violence features

np.load("/kaggle/working/violence_features.n py)

non violence features = np.load('/kaggle/working/non violence featur es.npy')

27/35

Creating labels

violence labels = np.zeros(len(violence_features)) non_violence_labels = np.ones(len(non_violence_features))

#Combining features and labels X= np.concatenate ([violence_features, non violence_features], axis=0) y= np.concatenate ([violence_labels, non violence _labels], axis=0)#Splitting data into training and

testing sets

X_train, X_test, y_train, y_test

train_test_split(x, y,

test size 0.2, random_state=32)

Reshaping data for LSTM input

X train reshaped =

X train.reshape((X_train.shape[0], 16, 2048))

X_test_reshaped =

X test.reshape((X_test.shape[0], 16, 2048))Accuracy: 

Develop algorithms and models capable of accurately detecting sensitive 

content categories such as violence, nudity, hate speech, and graphic imagery within 

videos, minimizing false positives and false negatives. 

Scalability: 

Design scalable solutions that can handle large volumes of video content in real-

time, ensuring timely detection and moderation of sensitive material across diverse 

online platforms. 

Privacy: 

Safeguard user privacy and data confidentiality during the content analysis 

process, adhering to privacy regulations and best practices to prevent unauthorized 

access or misuse of personal information. 

Accessibility: 

Ensure that sensitive content detection systems are accessible and inclusive, 

providing support for users with disabilities and diverse linguistic backgrounds, while 

minimizing biases and discriminatory outcome# Sensitive-Content-detection-in-Videos-
