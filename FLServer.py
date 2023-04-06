import glob
import sys
import socket
import os
import threading
import time
import numpy as np

# Create and insert path to Thrift libraries
directory = '/home/vkmg13/Documents/Thrift/thrift16/lib/py/build/lib*'
sys.path.insert(0, glob.glob(directory)[0])

# Import FLServer and Thrift files
from fl_support.FLServer import FLServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Flatten,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    Dropout,
)

def init_weights():
    model = Sequential()
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
    w = (model.get_weights())
    model.summary()
    timeQ = input('Press enter to continue')
    return w

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Hopital class stores weights for each class
class HospitalClient:
    def __init__(self, nameIn):
        self.name = nameIn
        self.hospital_weights = []
    
    def add_weights(self, new_weights):
        self.hospital_weights.append(new_weights)
    
    def get_hospital_weights(self):
        return self.hospital_weights
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class FLServerHandler:
    
    def __init__(self) -> None:
        # Weights of the server
        self.weights = init_weights()
        # Weights of individual clients
        self.client_weights = {}
        # Hospital tracker
        self.hospitals = {}
        # Number of clients
        self.num_clients = 0
        # Number of new sets obtained
        self.counter = 0
        # Number of times weights were aggregated
        self.version_number_tracker = 0
    
    # For clients to obtain weights
    # Note that these may be initialized or updated weights

    #######################################################################
    # Send weights to clients
    def send_first_layer(self):
        return self.weights[0]
    def send_second_layer(self):
        return self.weights[1]
    def send_third_layer(self):
        return self.weights[2]
    def send_fourth_layer(self):
        return self.weights[3]
    def send_fifth_layer(self):
        return self.weights[4]
    def send_sixth_layer(self):
        return self.weights[5]
    #######################################################################


    #######################################################################
    # Receive weights from clients
    def receive_first_layer(self, name, weights):
        self.hospitals[name] = HospitalClient(name)
        self.hospitals[name].add_weights(np.asarray(weights))
    def receive_second_layer(self, name, weights):
        self.hospitals[name].add_weights(np.asarray(weights))
    def receive_third_layer(self, name, weights):
        self.hospitals[name].add_weights(np.asarray(weights))
    def receive_fourth_layer(self, name, weights):
        self.hospitals[name].add_weights(np.asarray(weights))
    def receive_fifth_layer(self, name, weights):
        self.hospitals[name].add_weights(np.asarray(weights))
    def receive_sixth_layer(self, name, weights):
        self.hospitals[name].add_weights(np.asarray(weights))
        self.save_client_weights(name)

    def save_client_weights(self, name):
        if not name in self.client_weights:
            self.num_clients += 1
        self.client_weights[name] = self.hospitals[name].get_hospital_weights()
        self.counter += 1
        print('\n\nNew weights collected')
    #######################################################################
    
    def update_weights(self):
        # Aggregate weights
        self.weights = []
        print('Number of clients: '+ str(self.num_clients))
        if self.num_clients != 0:
            count = 0
            for values in self.client_weights.values():
                if count == 0:
                    self.weights.append(values[0]/self.num_clients)
                    self.weights.append(values[1]/self.num_clients)
                    self.weights.append(values[2]/self.num_clients)
                    self.weights.append(values[3]/self.num_clients)
                    self.weights.append(values[4]/self.num_clients)
                    self.weights.append(values[5]/self.num_clients)
                else:
                    self.weights[0] += values[0]/self.num_clients
                    self.weights[1] += values[1]/self.num_clients
                    self.weights[2] += values[2]/self.num_clients
                    self.weights[3] += values[3]/self.num_clients
                    self.weights[4] += values[4]/self.num_clients
                    self.weights[5] += values[5]/self.num_clients
                count += 1

            # Update times
            print(self.weights[0].shape)
            print(self.weights[1].shape)
            print(self.weights[2].shape)
            print(self.weights[3].shape)
            print(self.weights[4].shape)
            print(self.weights[5].shape)

            self.version_number_tracker += 1
            print('\n\n Aggregated weights...\n')    
            return self.version_number_tracker
        else:
            return 0
    
    def number_weights_collected(self):
        return self.counter
    
    def version_number(self):
        return self.version_number_tracker

    


if __name__ == '__main__':
    handler = FLServerHandler()
    processor = FLServer.Processor(handler)
    transport = TSocket.TServerSocket(host='localhost', port=9095)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    os.system('clear')
    print('\n')
    print("Server Port: " + str(9095))

    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    print('Server is starting...')
    server.serve()
    print('Server is off') 




