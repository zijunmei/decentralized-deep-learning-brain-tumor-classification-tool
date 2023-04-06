import glob
import sys
import socket
import os
import time


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Flatten,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Dropout,
)
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import random
import matplotlib.pyplot as plt

directory = '/home/vkmg13/Documents/Thrift/thrift16/lib/py/build/lib*'
print(os.getcwd())
sys.path.insert(0, glob.glob(directory)[0])

from fl_support.FLServer import FLServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def menu(name, trainp, validp, testp):
    choice = 1
    my_weights = []
    model = Sequential()
    trainModel = False
    """Initialize the training information"""
    model_name = name + ".h5"
    # Hospital A Directories
    
    train_path = trainp #Need to define
    valid_path = validp #Need to define
    test_path =  testp #Need to define
    train_batches = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                directory=train_path, target_size=(224, 224), batch_size=64, class_mode="sparse"
            )           
    valid_batches = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                directory=valid_path,
                target_size=(224, 224),
                batch_size=64,
                shuffle=False,
                class_mode="sparse",
            )
    test_batches = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
                directory=test_path,
                target_size=(224, 224),
                batch_size=64,
                shuffle=False,
                class_mode="sparse",
            )
    while(choice >= 1 and choice <= 5):
        
        cleanDisplay()
        menuDisplay(hospital_name)
        try:
            choice = int(input('Selection [1-4]: '))
        except:
            print('Selection is not a number')
            time.sleep(2)
            continue
        print()
        print('======================================')
        
        #################################################################
        # Ensure user made appropriate choice
        if(choice > 4 or choice < 1):
            print()
            print('Selection ' + str(choice) + ' is not available.')
            print()
            time.sleep(2)
            continue


        #################################################################
        # Train Model
        elif (choice == 1):
            # Obtain weights from server
            my_weights = []
            my_weights.append(np.asarray(client.send_first_layer()))
            print(my_weights[0].shape)
            my_weights.append(np.asarray(client.send_second_layer()))
            print(my_weights[1].shape)
            my_weights.append(np.asarray(client.send_third_layer()))
            print(my_weights[2].shape)
            my_weights.append(np.asarray(client.send_fourth_layer()))
            print(my_weights[3].shape)
            my_weights.append(np.asarray(client.send_fifth_layer()))
            print(my_weights[4].shape)
            my_weights.append(np.asarray(client.send_sixth_layer()))
            print(my_weights[5].shape)

            # TODO
            # 1. Train the model
            if not trainModel:
                model.add(
                    Conv2D(
                        input_shape=(224, 224, 3),
                        filters=16,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu",
                        name="Conv",
                    )
                )
                model.add(
                    Conv2D(filters=8, kernel_size=3, strides=1, padding="same", activation="relu")
                )
                model.add(MaxPool2D(pool_size=2, strides=2, padding="valid"))
                model.add(Flatten(name="Flatten"))
                model.add(Dense(4, activation="softmax", name="FC"))
                model.compile(
                    optimizer=Adam(lr=3e-4),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )
            model.set_weights(my_weights)
            monitor = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0,
                patience=3,
                verbose=0,
                mode="auto",
                baseline=None,
                restore_best_weights=True,
            )
            history = model.fit(
                x=train_batches,
                steps_per_epoch=len(train_batches),
                validation_data=valid_batches,
                validation_steps=len(valid_batches),
                epochs=2,
                verbose=1,
                callbacks=[monitor],
                )
            # Save weights for client
            
            my_weights = model.get_weights()

            # 2. Print accuracy of model
            testModel(model,model_name, test_path, test_batches)
            timeQ = input('Press enter to continue')
            # 3. Update trainModel to True
            trainModel = True
        #################################################################
        # Send Model
        elif (choice == 2):
            # If the model has not been trained print a statement
            # and continue.
            # Otherwise send model to server
            if trainModel:
                client.receive_first_layer(name, my_weights[0])
                client.receive_second_layer(name, my_weights[1])
                client.receive_third_layer(name, my_weights[2])
                client.receive_fourth_layer(name, my_weights[3])
                client.receive_fifth_layer(name, my_weights[4])
                client.receive_sixth_layer(name, my_weights[5])
            else:
                print('Model has not been trained')
                time.sleep(2)
                continue
            

        #################################################################
        # Get Model
        elif(choice == 3):
            my_weights = []
            my_weights.append(np.asarray(client.send_first_layer()))
            print(my_weights[0].shape)
            my_weights.append(np.asarray(client.send_second_layer()))
            print(my_weights[1].shape)
            my_weights.append(np.asarray(client.send_third_layer()))
            print(my_weights[2].shape)
            my_weights.append(np.asarray(client.send_fourth_layer()))
            print(my_weights[3].shape)
            my_weights.append(np.asarray(client.send_fifth_layer()))
            print(my_weights[4].shape)
            my_weights.append(np.asarray(client.send_sixth_layer()))
            print(my_weights[5].shape)
            # TODO
            # Test updated model for accuracy
            new_model = keras.models.load_model(model_name)
            new_model.set_weights(my_weights)
            testModel(new_model,model_name, test_path, test_batches)
            timeQ = input('Press enter to continue')
                    
            
        #################################################################
        # Log off
        elif(choice == 4):
            print('*****************************')
            print('\n\t\tLogged Off\n')
            print('*****************************')
            transport.close()
            break

# Print model accuracy and plots
# Use this method after training model and after receiving model
# (i.e., option 1 and 3)
def testModel(model,model_name,test_path,test_batches):
    preds = model.evaluate(test_batches)
    print("Test Accuracy is " + str(preds[1] * 100) + "%")
    model.save(model_name)
    print("Model Saved!")
    
def menuDisplay(hospital_name):
    print('\n')
    print('\t' + hospital_name)
    print('\n')
    print('======================================')
    print('Select from following options:')
    print('1. Train Model')
    print('2. Send Weights to Server')
    print('3. Get Weights from Server')
    print('4. Log Out')
    print()

def cleanDisplay():
    os.system('clear')
    print('\n')


if __name__=='__main__':
    filename = input('Enter settings file name: ')
    file = open(filename,'r')
    ipaddr = file.readline().replace('\n', '')
    trainp = file.readline().replace('\n', '')
    validp = file.readline().replace('\n', '')
    testp = file.readline().replace('\n', '')
    # Create Socket    
    print("Server Port: " + str(9095))
    transport = TSocket.TSocket('localhost', 9095)
    # Buffering Transport
    transport = TTransport.TBufferedTransport(transport)
    # Create Protocol
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    # Create Client
    client = FLServer.Client(protocol)
    # Connect to ChatServer
    transport.open()
    os.system('clear')
    hospital_name = input('Enter hospital name: ')
    menu(hospital_name, trainp, validp, testp)