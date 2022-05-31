import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

def format_example(image, label):
    """
    Returns an image that is reshaped to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

if __name__ == "__main__":
    ###### Preliminary: Build CNN on our own
    ### Load and split dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    ### Preprocess data: Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Uncomment the following to visualise an image
    #IMG_INDEX = 7  # change this to look at other images
    #plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
    #plt.show()

    # Build CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Adding dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))

    # Training
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=4, 
                        validation_data=(test_images, test_labels))
                        
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(test_acc)

    ### Data augmentation
    # Creates a data generator object that transforms images
    datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

    # Pick an image to transform
    test_img = train_images[20]
    img = image.img_to_array(test_img)  # convert image to numpy arry
    img = img.reshape((1,) + img.shape)  # reshape image

    i = 0
    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
        plt.figure(i)
        plot = plt.imshow(image.img_to_array(batch[0]))
        i += 1
        if i > 4:  # show 4 images
            break
    plt.show()

    ###### Build CNN on a pretrained model
    keras = tf.keras
    tfds.disable_progress_bar()

    ### Split the data manually into 80% training, 10% testing, 10% validation
    (raw_train, raw_validation, raw_test), metadata = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels
    # Display 2 images from the dataset
    for image, label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

    ### Preprocess data so that images are of same size
    IMG_SIZE = 160 # All images will be resized to 160x160
    train = raw_train.map(format_example)
    validation = raw_validation.map(format_example)
    test = raw_test.map(format_example)

    # Quick look at our images
    for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

    # Shuffle and batch the images
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    validation_batches = validation.batch(BATCH_SIZE)
    test_batches = test.batch(BATCH_SIZE)

    # Quick look at original images VS new images
    for img, label in raw_train.take(2):
    print("Original shape:", img.shape)

    for img, label in train.take(2):
    print("New shape:", img.shape)

    ### Picking a pre-trained model (MobileNet V2)
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    for image, _ in train_batches.take(1):
        pass
    feature_batch = base_model(image)
    print(feature_batch.shape)

    # Freeze the base to not change the convolutional base
    base_model.trainable = False

    # Adding classifier
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Adding prediction layer
    prediction_layer = keras.layers.Dense(1)

    # Combining all the layers
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    ### Training model
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    
    # Do pre-training evaluation: We evaluate the model right now to see how it does before training it on our new images
    initial_epochs = 3
    validation_steps=20
    loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

    history = model.fit(train_batches,
                        epochs=initial_epochs,
                        validation_data=validation_batches)
    acc = history.history['accuracy']
    print(acc)
    model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
    new_model = tf.keras.models.load_model('dogs_vs_cats.h5')
