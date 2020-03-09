import numpy as np
from data_prep import get_sonnets_from_file, parse_syllable_list
from HMM import HiddenMarkovModel, unsupervised_HMM

if __name__ == '__main__':
    # Load the data.
    sonnet_list = get_sonnets_from_file('shakespeare.txt')
    syls, w2i, i2w = parse_syllable_list('Syllable_dictionary.txt')

    # Convert all of the words in the sonnets to indices given by w2i.
    index_sonnets = [[[w2i[word] for word in line if word in w2i]
                                 for line in sonnet]
                                 for sonnet in sonnet_list]
    
    # Generate the data as one long list of lines.
    X = []
    for sonnet in index_sonnets:
        for line in sonnet:
            X.append(line)

    # Get the maximum word index is the number of words - 1.
    m = max([max(line) for line in X])

    # Train the Hidden Markov Model.
    hmm = unsupervised_HMM(X, 20, 1)

    # Make Shakespeare look like a chump.
    for _ in range(14):
        emission_list = hmm.generate_emission()[0]
        print(' '.join(map(lambda idx: i2w[idx], emission_list)))