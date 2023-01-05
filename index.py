from sklearn.feature_extraction import image
import sklearn
from sklearn import svm
import numpy as np
import cv2
import os

def extract_features(image):
  # Convert the image to the HSV color space
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Calculate the color histogram
  histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

  # Normalize the histogram
  cv2.normalize(histogram, histogram)

  return histogram.flatten()

NAFiles = []
for file in os.listdir('/NA/'):
    NAFiles.append(cv2.imread(file))

SAFiles = []
for file in os.listdir('/SA/'):
    SAFiles.append(cv2.imread(file))

ASFiles = []
for file in os.listdir('/AS/'):
    ASFiles.append(cv2.imread(file))

for file in range(len(NAFiles)):
    NAFiles[file] = cv2.resize(NAFiles[file], (300, 300))

for file in range(len(SAFiles)):
    SAFiles[file] = cv2.resize(SAFiles[file], (300, 300))

for file in range(len(ASFiles)):
    ASFiles[file] = cv2.resize(ASFiles[file], (300, 300))

# Convert the images to numpy arrays
for file in range(len(NAFiles)):
    NAFiles[file] = np.array(NAFiles[file])

for file in range(len(SAFiles)):
    SAFiles[file] = np.array(SAFiles[file])

for file in range(len(ASFiles)):
    ASFiles[file] = np.array(ASFiles[file])

# Create a numpy array of the images
images = np.array([])

for file in NAFiles:
    images.append(file)

for file in SAFiles:
    images.append(file)

for file in ASFiles:
    images.append(file)

print(images)
# Create a list of labels corresponding to each image
labels = np.array([])


# Extract features from the images
features = []
for i in range(len(images)):
  features.append(extract_features(images[i]))
features = np.array(features)

# Create a classifier and train it using the extracted features and labels
classifier = sklearn.svm.SVC()
classifier.fit(features, labels)

# Load the new image and extract features from it
new_image = cv2.imread('image4.png')
new_image = cv2.resize(new_image, (300, 300))
new_image = np.array(new_image)
new_features = extract_features(new_image)
new_features = new_features.reshape(1, -1)
# Use the trained classifier to predict the label for the new image
prediction = classifier.predict(new_features)
print(prediction)
