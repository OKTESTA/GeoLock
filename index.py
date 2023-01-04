from sklearn.feature_extraction import image
import numpy as np

# Load the images as a numpy array
image1 = image.imread('image1.jpg')
image2 = image.imread('image2.jpg')
image3 = image.imread('image3.jpg')
images = np.array([image1, image2, image3])

# Create a list of labels corresponding to each image
labels = np.array(['cat', 'dog', 'bird'])

# Extract features from the images
features = []
for i in range(len(images)):
  features.append(image.extract_features(images[i]))
features = np.array(features)

# Create a classifier and train it using the extracted features and labels
classifier = sklearn.svm.SVC()
classifier.fit(features, labels)

# Load the new image and extract features from it
new_image = image.imread('image4.jpg')
new_features = image.extract_features(new_image)

# Use the trained classifier to predict the label for the new image
prediction = classifier.predict(new_features)
print(prediction)
