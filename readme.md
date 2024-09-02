# Mauritanian License Plate Recognition Challenge

This repository contains the code and steps used for the Mauritanian License Plate Recognition Challenge, a Kaggle competition focused on recognizing and processing license plate characters from images. The goal is to preprocess the images, encode the license plate characters, and build a Convolutional Neural Network (CNN) model to accurately predict the characters.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Character Encoding](#character-encoding)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)



## Project Overview

The goal of this project is to build a model that can accurately recognize characters from license plate images. The project involves data preprocessing, character encoding, and the development of a Convolutional Neural Network (CNN) model to predict license plate characters.

## Dataset

The dataset used in this project consists of images of license plates from Mauritania. Each image is labeled with the corresponding license plate number in a CSV file.

- *train_labels.csv*: Contains the image ID and the corresponding license plate number.

Example:

| img_id | plate_number |
|--------|--------------|
| img_1  | 8630AB06     |
| img_10 | 5115AM00     |

## Preprocessing

The preprocessing steps involve loading the images, resizing them to a uniform size, and normalizing the pixel values. The preprocessed images are then saved in a new directory and zipped for easy access.

### Example of Preprocessing Code

python
def preprocess_image(img_path, target_size=(416, 416)):
    # Load the image
    img = cv2.imread(img_path)
    
    # Resize the image
    img_resized = cv2.resize(img, target_size)
    
    # Normalize the image (scale pixel values to [0, 1])
    img_normalized = img_resized / 255.0
    
    return img_normalized

# Preprocess and save all images
for img_id in train_labels['img_id']:
    img_path = os.path.join(img_dir, f'{img_id}.jpg')
    preprocessed_img = preprocess_image(img_path)

    if preprocessed_img is not None:
        output_path = os.path.join(output_dir, f'{img_id}_preprocessed.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR))


## Character Encoding

The license plate characters are encoded into numerical values to prepare them for training the model. A character-to-integer mapping is created, and each license plate number is converted into a sequence of integers.

### Example of Character Encoding Code

python
characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
char_to_num = {char: i for i, char in enumerate(characters)}

def encode_plate(plate):
    return [char_to_num[char] for char in plate]

encoded_labels = train_labels['plate_number'].apply(encode_plate)
encoded_labels = np.array(encoded_labels)


## Model Architecture

A Convolutional Neural Network (CNN) model is constructed with multiple convolutional blocks followed by fully connected layers. The model is designed to take preprocessed images as input and output the predicted license plate characters.

### Example of Model Architecture Code

python
model = Sequential()

# First Convolutional Block
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(416, 416, 3)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Additional convolutional blocks...


## Training

The dataset is split into training and validation sets. The model is trained using the training set, and its performance is evaluated on the validation set.

### Example of Training Code

python
X_train, X_val, y_train, y_val = train_test_split(preprocessed_images, one_hot_labels, test_size=0.2, random_state=42)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)


## Results

After training, the model's performance is evaluated on the validation set, and the accuracy and loss are tracked throughout the training process.

## Usage

To use the code in this repository:

1. Clone the repository.
2. Install the required dependencies.
3. Run the preprocessing steps.
4. Train the model using the provided training script.