import glob
import sys
import socket
import os
import time

directory = '/home/vkmg13/Documents/Thrift/thrift16/lib/py/build/lib*'
print(os.getcwd())
sys.path.insert(0, glob.glob(directory)[0])

from fl_support.FLServer import FLServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def menu():
    choice = 1
    while(choice >= 1 and choice <= 5):
        
        cleanDisplay()
        menuDisplay()
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
        # Aggregate Weights
        elif (choice == 1):
            version = client.update_weights()
            if version == 0:
                print('\n\n**************************************\n')
                print('No weights have been collected')
                print('\n**************************************\n')
                timeQ = input('Press enter to continue')
            else:
                print('\n\n**************************************\n')
                print('New version number of aggregated model: ' + str(version)  )
                print('\n**************************************\n')
                timeQ = input('Press enter to continue')
        #################################################################
        # Check Version Number
        elif (choice == 2):
            version = client.version_number()
            print('\n\n**************************************\n')
            print('Version number of aggregated model: ' + str(version)  )
            print('\n**************************************\n')
            timeQ = input('Press enter to continue')

        #################################################################
        # Get number of sets of weights collected
        elif(choice == 3):
            counter = client.number_weights_collected()
            print('\n\n**************************************\n')
            print('The weight sets collected thus far is: ' + str(counter)  )                 
            print('\n**************************************\n')
            timeQ = input('Press enter to continue')
        #################################################################
        # Log off
        elif(choice == 4):
            print('*****************************')
            print('\n\tServer Shut Down\n')
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
    
def menuDisplay():
    print('\n')
    print('\tADMIN')
    print('\n')
    print('======================================')
    print('Select from following options:')
    print('1. Aggregate Weights')
    print('2. Check Version Number')
    print('3. Get number of sets of weights collected')
    print('4. Log Out')
    print()

def cleanDisplay():
    os.system('clear')
    print('\n')


if __name__=='__main__':
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

    menu()