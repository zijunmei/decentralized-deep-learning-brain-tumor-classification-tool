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
