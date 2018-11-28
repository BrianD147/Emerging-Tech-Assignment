# Note: A command has been included at the top of the digit-recognition notebook to easily run this code

# Imports
import keras as kr
import numpy as np
import sklearn.preprocessing as pre
import gzip

# Build the neural network

# Initialise the network
model = kr.models.Sequential()

# Adding layers, 784 input neurons, 1000 hidden layer neurons, and an additional 400 hidden layer neurons
model.add(kr.layers.Dense(units=1000, activation='linear', input_dim=784))
model.add(kr.layers.Dense(units=400, activation='relu'))

# Add 10 output neurons, one for each possible digit the image could be containing (0-9)
model.add(kr.layers.Dense(units=10, activation='softmax'))

# Compile the network graph
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Read in all the files
with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
    
with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = f.read()
    
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    training_images = f.read()
    
with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    training_labels = f.read()

# Read all the files into memory
training_images = ~np.array(list(training_images[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
training_labels =  np.array(list(training_labels[8:])).astype(np.uint8)

test_images = ~np.array(list(test_images[16:])).reshape(10000, 784).astype(np.uint8) / 255.0
test_labels = np.array(list(test_labels[8:])).astype(np.uint8)

# Flatten the array to corrispond with the 784 input neurons
neuron_inputs = training_images.reshape(60000, 784)

# Set up the labels

# Encode the data into binary values
encoder = pre.LabelBinarizer()
encoder.fit(training_labels)
neuron_outputs = encoder.transform(training_labels)

# Ask user if they would like to run a training session, or use the model in it's current state
print()
print()
print("Would you like to run a training session? (Note: choosing no will run the model in it's current state)")
option = input("y/n : ")
if option == 'y':
    # Train the model
    model.fit(neuron_inputs, neuron_outputs, epochs=20, batch_size=100)
    # Save the model
    model.save("data/model.h5")
elif option == 'n':
    # Load the model
    model.load_weights("data/model.h5")
else:
    print("*Invalid option selected*")
    print("**Model will not produce accurate predictions due to lack of training**")
    
print()
print()
print("1 - Test the model against random digits from the dataset")
print("2 - Test the model against your own image of a digit")
option = input("1/2 : ")
if option == '1':
    from random import randint
    
    for i in range(20): #Run 20 tests
        print("=================================")
        randIndex = randint(0, 9999) #Get a random index to pull an image from
        test = model.predict(test_images[randIndex:randIndex+1]) #Pull the image from the dataset
        result = test.argmax(axis=1) #Set result to the highest array value
        print("The actual number: =>> ", test_labels[randIndex:randIndex+1])
        print("The network reads: =>> ", result)
        print("=================================")
elif option == '2':
    # Imports
    import tkinter as tk
    from tkinter import filedialog
    from keras.preprocessing import image
    
    # Set root and hide root window using .withdraw()
    root = tk.Tk()
    root.withdraw()
    
    # Prompt filename and load image
    file_path = filedialog.askopenfilename()
    user_image = image.load_img(path=file_path,color_mode = "grayscale",target_size=(28,28,1))
    
    # Flatten image into single dimentional array
    flattened_image = np.array(list(image.img_to_array(user_image))).reshape(1, 784).astype(np.uint8) / 255.0
    
    test = model.predict(flattened_image) #Allow network to read the image
    result = test.argmax(axis=1) #Set result to the highest array value
    
    print("=================================")
    print("The network reads: =>> ", result)
    print("=================================")
else:
    print("*Invalid option selected*")
    print("**Exiting without running model**")
    

