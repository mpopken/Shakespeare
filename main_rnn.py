'''
This file contains the main implementation of the sonnet generation via RNN. 
Here, we load the data from the text file, process it, train an
RNN on the data, and generate a new sonnet which is printed to the console.
'''

import numpy as np
import nltk
from data_prep import *
from RNN import *

nltk.download('cmudict')

if __name__ == '__main__':
     # Load the data.
    sonnet_list = get_sonnets_from_file('shakespeare.txt')
    rnn = setup_rnn(sonnet_list)
    temperatures = [1.5, 0.75, 0.25]
    
    for temp in temperatures:
        print("\nTEMPERATURE = ", temp)
        print('#####################')
        sequence = "shall i compare thee to a summer's day?\n"
        print(sequence, end='')
        
        for i in range(200): # generate 200 characters
            res = rnn.predict(sequence, temperature=temp)
            print(res, end='')
            sequence += res
            sequence = sequence[1:]