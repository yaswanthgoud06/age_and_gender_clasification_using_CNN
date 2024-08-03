import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import Sequence

class ImageDataGenerator(Sequence):
    def __init__(self, image_filenames, ages, genders, batch_size):
        self.image_filenames = image_filenames
        self.ages = ages
        self.genders = genders
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_age = self.ages[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_gender = self.genders[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            np.array(Image.open(file_name).resize((64, 64))) / 255.0
            for file_name in batch_x
        ]), {'age_output': batch_y_age, 'gender_output': batch_y_gender}

def load_utkface_dataset(data_dir):
    image_filenames = []
    ages = []
    genders = []
    
    print(f"Attempting to load data from: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")
    
    files = os.listdir(data_dir)
    print(f"Total files found: {len(files)}")
    
    for filename in files:
        print(f"Processing file: {filename}")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    age, gender = int(parts[0]), int(parts[1])
                    image_filenames.append(os.path.join(data_dir, filename))
                    ages.append(age)
                    genders.append(gender)
                    print(f"  Valid image: {filename}, Age: {age}, Gender: {gender}")
                except ValueError:
                    print(f"  Skipping file {filename} due to invalid age or gender format")
            else:
                print(f"  Skipping file {filename} due to unexpected format")
        else:
            print(f"  Skipping file {filename} as it's not a recognized image format")
    
    print(f"Valid images found: {len(image_filenames)}")
    
    if not image_filenames:
        raise ValueError("No valid images found in the specified directory.")
    
    return image_filenames, np.array(ages), np.array(genders)

def create_model(input_shape):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    age_output = Dense(1, name='age_output')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)
    
    model = Model(inputs=inputs, outputs=[age_output, gender_output])
    return model

def main():
    data_dir = r'C:\Users\yaswa\OneDrive\Desktop\proglint\data\UTKFace'
    
    try:
        image_filenames, ages, genders = load_utkface_dataset(data_dir)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return

    print(f"Dataset loaded. Images: {len(image_filenames)}, Ages: {len(ages)}, Genders: {len(genders)}")

    X_train, X_val, y_age_train, y_age_val, y_gender_train, y_gender_val = train_test_split(
        image_filenames, ages, genders, test_size=0.2, random_state=42
    )

    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    train_generator = ImageDataGenerator(X_train, y_age_train, y_gender_train, batch_size=32)
    val_generator = ImageDataGenerator(X_val, y_age_val, y_gender_val, batch_size=32)

    input_shape = (64, 64, 3)
    model = create_model(input_shape)
    model.compile(
        optimizer='adam',
        loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
        metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=40
    )

    model.save('age_gender_model.h5')

    print("Training completed. Model saved as 'age_gender_model.h5'")

if __name__ == '__main__':
    main()