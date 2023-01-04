from sklearn.feature_extraction import image
import sklearn from sklearn import svm
import numpy as np
import cv2

def extract_features(image):
  # Convert the image to the HSV color space
  image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Calculate the color histogram
  histogram = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

  # Normalize the histogram
  cv2.normalize(histogram, histogram)

  return histogram.flatten()

# Load the images as a numpy array
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')
image3 = cv2.imread('image3.png')
if image1 is None:
  print("Failed to load image1.jpg")
if image2 is None:
  print("Failed to load image2.jpg")
if image3 is None:
  print("Failed")

# Make sure all images have the same shape
image1 = cv2.resize(image1, (300, 300))
image2 = cv2.resize(image2, (300, 300))
image3 = cv2.resize(image3, (300, 300))

# Convert the images to numpy arrays
image1 = np.array(image1)
image2 = np.array(image2)
image3 = np.array(image3)

# Create a numpy array of the images
images = np.array([image1, image2, image3])

# Create a list of labels corresponding to each image
labels = np.array(['NA', 'SA', 'AS'])

# Extract features from the images
features = []
for i in range(len(images)):
  features.append(extract_features(images[i]))
features = np.array(features)

# Create a classifier and train it using the extracted features and labels
classifier = sklearn.svm.SVC()
classifier.fit(features, labels)

# Load the new image and extract features from it
new_image = image.imread('image1.png')
new_features = extract_features(new_image)

# Use the trained classifier to predict the label for the new image
prediction = classifier.predict(new_features)
print(prediction)
