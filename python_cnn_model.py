# Import statements for tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Constants
input_shape = (150, 150, 3)  # Input image dimensions
num_classes = 120  # Number of classes in the dataset

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Assuming you have your training and validation data prepared as train_data and validation_data
# Replace 'train_data' and 'validation_data' with your actual datasets

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories containing training and validation data
train_dir = 'training_data'
validation_dir = 'validation_data'

# Data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


# Define number of epochs and batch size
epochs = 2 
batch_size = 128 

# Flow training images in batches using train_datagen
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'sparse' for sparse_categorical_crossentropy loss
)

print(train_data)


# Flow validation images in batches using validation_datagen
validation_data = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='sparse'  # Use 'sparse' for sparse_categorical_crossentropy loss
)


#set the data

# Fitting the model to the dataset
history = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

