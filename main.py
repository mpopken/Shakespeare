'''
This file contains the main implementation of the sonnet generation via Hidden
Markov Models. Here, we load the data from the text file, process it, train an
HMM on the data, and generate a new sonnet which is printed to the console.
'''

import numpy as np
import nltk
from data_prep import *
from HMM import HiddenMarkovModel, unsupervised_HMM

nltk.download('cmudict')

if __name__ == '__main__':
    # Load the data.
    sonnet_list = get_sonnets_from_file('shakespeare.txt')
    syls, w2i, i2w = parse_syllable_list('Syllable_dictionary.txt')

    # Convert all of the words in the sonnets to indices given by w2i.
    index_sonnets = [[[w2i[word] for word in line if word in w2i]
                                 for line in sonnet]
                                 for sonnet in sonnet_list]
    
    # Generate the data as one long list of lines. We have to generate the
    # lines in reverse because it is easier to make the first word rhyme.
    X = []
    for sonnet in index_sonnets:
        for line in sonnet:
            X.append(list(reversed(line)))

    # Get the maximum word index is the number of words - 1.
    m = max([max(line) for line in X])

    # Train the Hidden Markov Model.
    hmm = unsupervised_HMM(X, 20, 1)

    ## Generate a sonnet using the HMM
    output = []
    last = []
    num_lines = 14
    rhyme_lines = [3, 4, 7, 8, 11, 12, 14]
    for i in range(1, num_lines + 1):
        if i in rhyme_lines:
            rhyme_bank = [w2i[word] for word in rhyme(i2w[last[i - 3]], w2i.keys())]
            lst = hmm.generate_emission(10, rhyme_bank)[0]
        else:
            lst = hmm.generate_emission(10)[0]
            
        output.append(' '.join(map(lambda idx: i2w[idx], reversed(lst))))
        last.append(lst[0])

    ## Present the masterpiece.
    for line in output:
        print(line)