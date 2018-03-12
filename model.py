import csv
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2


def loadData(directory, skipHeader=True):
	"""
	The data directory needs to have a csv driving data logfile and
	a directory containing left, center and right images
	"""
	# Load the csv file
	lines = []
	with open(directory + '/driving_log.csv', 'r') as f:
		csvreader = csv.reader(f)
		if skipHeader:
			next(csvreader, None)
		for line in csvreader:
			lines.append(line)
			
	measurements, centerImgs, leftImgs, rightImgs = [], [], [], []
	
	# We now have the measurements loaded
	# Steering angles are in the fourth column.
	# Path to center, left and right are in the first three columns resp
	for line in lines:
		# Skip very low speeds
		if float(line[6]) < 1:
			continue
		measurements.append(float(line[3]))
		centerImgs.append(directory + '/' + line[0].strip())
		leftImgs.append(directory + '/' + line[1].strip())
		rightImgs.append(directory + '/' + line[2].strip())
	
	return (measurements, centerImgs, leftImgs, rightImgs)

def loadDataSingle(directory, skipHeader=True, offsetFactor=0.2):
	"""
	This method loads and combines the left, right and center image data
	The angles are offset by a factor.
	The data directory needs to have a csv driving data logfile and
	a directory containing left, center and right images
	"""
	# Load the csv file
	lines = []
	with open(directory + '/driving_log.csv', 'r') as f:
		csvreader = csv.reader(f)
		if skipHeader:
			next(csvreader, None)
		for line in csvreader:
			lines.append(line)
			
	measurements, imgs = [], []
	
	# We now have the measurements loaded
	# Steering angles are in the fourth column.
	# Path to center, left and right are in the first three columns resp
	for line in lines:
		# Skip very low speeds
		if float(line[6]) < 1:
			continue
		angle = float(line[3])
		# Center image
		imgs.append(directory + '/' + line[0].strip())
		measurements.append(angle)
		# Left image
		imgs.append(directory + '/' + line[1].strip())
		measurements.append(angle + offsetFactor)
		# Right image
		imgs.append(directory + '/' + line[2].strip())
		measurements.append(angle - offsetFactor)
	
	return (measurements, imgs)

def generator(imgPaths, measurements, batch_size=128, flip=False):
	"""
	Generator for the data
	"""
	num = len(imgPaths)
	imgPaths, measurements = shuffle(imgPaths, measurements)
	
	imgs, angles = [], []
	while True:
		for i in range(num):
			img = cv2.imread(imgPaths[i])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			angle = measurements[i]
			
			imgs.append(img)
			angles.append(angle)
			
			# Flip horizontally only if the steering angle is significant
			if abs(angle) > 0.1 and flip:
				img = cv2.flip(img, 1)
				angle *= -1
				
				imgs.append(img)
				angles.append(angle)
			
			if len(imgs) >= batch_size:
				imgs = np.array(imgs[:batch_size])
				angles = np.array(angles[:batch_size])
				
				yield (imgs, angles)
				imgs, angles = [], []
				imgPaths, measurements = shuffle(imgPaths, measurements)



def createModel():
	"""
	NVIDIA's End-to-End training model
	"""
	model = Sequential()
	
	# Normalization layer
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	
	# Additional cropping layer to speed up training and testing
	model.add(Cropping2D(cropping=((50,20), (0,0))))
	
	# Convolutional layers
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	
	# Flatten layer
	model.add(Flatten())
	
	# Fully connected layers
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	
	return model


print('Loading data...')
measurements, imgs = loadDataSingle('./data/data')
new_meas, new_imgs = loadDataSingle('./data/track1/recovery')
measurements.extend(new_meas)
imgs.extend(new_imgs)
#print(len(new_meas))
#exit(0)
print('Done loading data...')


# Split the dataset into train and validation
rand_state = np.random.randint(0, 100)
imgs_train, imgs_valid, measure_train, measure_valid = train_test_split(imgs, measurements,
																	test_size=0.2,
																	random_state=rand_state)
print('Done splitting dataset into training and validation')

# Get the generators for each
train_generator = generator(imgs_train, measure_train, flip=True)
valid_generator = generator(imgs_valid, measure_valid)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.models import load_model

loadModel = False
modelFile = 'model.h5'

# Model init
if loadModel:
	print('Loading model from ', modelFile)
	model = load_model(modelFile)
else:
	print('Creating and compiling the model')
	model = createModel()
	model.compile(loss='mse', optimizer='adam')

print('Training the model....')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(imgs_train),
									validation_data=valid_generator,
									nb_val_samples=len(imgs_valid), nb_epoch=10, verbose=1)


# Save the model
model.save(modelFile)
print('model saved to', modelFile)











