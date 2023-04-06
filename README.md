# decentralized-deep-learning-brain-tumor-classification-tool
Develop a decentralized deep learning braintumor classification tool for medical practitioners.

1. FLServerHelper.py <br>
This Python script is a command-line interface for the server-side of a Federated Learning system, where multiple clients share their local model updates to improve a global model without sharing their raw data. The script imports necessary libraries, sets up the connection to the server, and provides a menu to interact with the server.<br>

Here's a breakdown of the code:<br>

Import necessary libraries such as glob, sys, socket, os, and time.<br>
Set the 'directory' variable to the path of the Thrift library.<br>
Print the current working directory and insert the Thrift library path to the system path.<br>
Import necessary classes and functions from the Thrift library.<br>
Define the 'menu()' function, which is the main loop for user interaction. Users can choose from the following options:<br>
a. Aggregate Weights: Update the global model by aggregating the weights from all clients.<br>
b. Check Version Number: Check the current version number of the aggregated model.<br>
c. Get number of sets of weights collected: Check how many weight sets have been collected so far.<br>
d. Log Out: Close the connection to the server and exit the program.<br>
Define the 'testModel()' function, which is not used in this script but can be used to print the model's accuracy and save the model after training.<br>
Define the 'menuDisplay()' function, which displays the menu options to the user.<br>
Define the 'cleanDisplay()' function, which clears the terminal screen and prints a newline character.<br>
The 'if name=='main':' block sets up the connection to the server and calls the 'menu()' function to start user interaction.<br>

2. FLClient.py <br> 
This Python script is a client-side implementation for a Federated Learning system, where the client trains a model on its local data, sends model updates to the server, and receives aggregated weights from the server. The script imports necessary libraries, sets up the connection to the server, and provides a menu for user interaction. <br>

Here's a breakdown of the code:<br>

Import necessary libraries, including TensorFlow, Keras, and other dependencies for image processing, and model training.<br>
Set the 'directory' variable to the path of the Thrift library.<br>
Print the current working directory and insert the Thrift library path to the system path.<br>
Import necessary classes and functions from the Thrift library.<br>
Define the 'menu()' function, which is the main loop for user interaction. Users can choose from the following options:<br>
a. Train Model: Train the local model using the data available at the client-side.<br>
b. Send Weights to Server: Send the local model weights to the server.<br>
c. Get Weights from Server: Retrieve the aggregated weights from the server.<br>
d. Log Out: Close the connection to the server and exit the program.<br>
Define the 'testModel()' function to evaluate the model's accuracy and save the model.<br>
Define the 'menuDisplay()' function, which displays the menu options to the user.<br>
Define the 'cleanDisplay()' function, which clears the terminal screen and prints a newline character.<br>
The 'if name=='main':' block takes user input for settings file name and hospital name, sets up the connection to the server, and calls the 'menu()' function to start user interaction.<br> 

3. FLServer.py<br>
This code sets up a federated learning server using the Thrift framework. It also includes TensorFlow for training deep learning models. The code has the following key components:<br>

Import necessary libraries: glob, sys, socket, os, threading, time, numpy, thrift, and tensorflow.<br>

Define a function init_weights() to create a convolutional neural network model and return its weights. The network consists of two convolutional layers followed by a max-pooling layer, a flatten layer, and a dense layer with softmax activation. The model is compiled using the Adam optimizer, with a sparse categorical crossentropy loss function and an accuracy metric.<br>

Define a HospitalClient class to store weights for each client (hospital). The class has an add_weights() method to add new weights and a get_hospital_weights() method to return the stored weights.<br>

Define the FLServerHandler class to handle server operations, including:<br>

Initializing the server's weights<br>
Storing client weights<br>
Keeping track of hospital clients<br>
Sending and receiving weights to/from clients<br>
Aggregating client weights to update the server's weights<br>
Tracking the number of times the server has aggregated client weights<br>
The if __name__ == '__main__': block creates an instance of the FLServerHandler class, sets up a Thrift server, and starts the server. The server listens on port 9095 and uses a threaded server model, allowing it to handle multiple client connections concurrently.<br>

The primary goal of this code is to set up a federated learning server that can distribute model weights to clients (hospitals) and receive updated weights back from them after training. The server then aggregates the weights from all clients and updates its global model.<br>

4. FLServer.thrift<br>
This code defines a Thrift service named FLServer for a federated learning server. The service contains methods for clients to request updated weights, send new weights to the server, and get information about the server's status.<br>

The service has the following methods:<br>

send_first_layer(), send_second_layer(), send_third_layer(), send_fourth_layer(), send_fifth_layer(), send_sixth_layer(): These methods return the corresponding layers of the server's model weights as nested lists of doubles. Clients call these methods to obtain the latest model weights from the server.<br>

receive_first_layer(), receive_second_layer(), receive_third_layer(), receive_fourth_layer(), receive_fifth_layer(), receive_sixth_layer(): These methods take a client name (string) and the corresponding layers of the client's model weights as nested lists of doubles. Clients call these methods to send their updated model weights to the server.<br>

number_weights_collected(): This method returns the number of new weight sets collected by the server as an int64 value.<br>

version_number(): This method returns the current version number of the server's model, which represents the number of times the server has aggregated client weights, as an int64 value.<br>

update_weights(): This method updates the server's model weights by aggregating the client weights and returns the new version number as an int64 value.<br>

The FLServer service definition provides the necessary interface for clients and the server to interact in a federated learning environment. Clients can obtain the latest model weights, send their updated weights to the server, and get information about the server's status. The server can receive client weights, aggregate them, and update its global model.<br>
