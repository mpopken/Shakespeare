import numpy as np
from data_prep import get_sonnets_from_file, parse_syllable_list
from HMM import HiddenMarkovModel, unsupervised_HMM

def make_line(hmm, word_to_rhyme=None):
    # Generate a line from the model (reversed to ensure last word rhymes).
    emission_list = hmm.generate_emission(10, rhymes_with[word_to_rhyme])[0]

    # Convert the index list to words, and make it a sentence.
    return ' '.join(map(lambda idx: i2w[idx], reversed(emission_list)))


def create_rhyming_dictionary(idx_list):
    # Initialise the rhyming dictionary.
    rhymes_with = {}
    rhymes_with[None] = None

    # Go through each word (index) and get the words (indices) that rhyme.
    for i in idx_list:
        rhymes_with[i] = []
        word = i2w[i]
        for j in idx_list:
            # Words don't rhyme with themselves.
            if i == j:
                continue
            # Words rhyme if their last two letters are the same.
            if word[-min(2, len(word)):] == i2w[j][-min(2, len(word)):]:
                rhymes_with[i].append(j)
        # assert len(rhymes_with[i]) > 0, f'{i2w[i]} has no rhymes!'
    
    return rhymes_with

if __name__ == '__main__':
    # Load the data.
    sonnet_list = get_sonnets_from_file('shakespeare.txt')
    syls, w2i, i2w = parse_syllable_list('Syllable_dictionary.txt')

    # Create the rhyming dictionary.
    rhymes_with = create_rhyming_dictionary(list(i2w.keys()))

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

    output = []

    ## Make Shakespeare look like a chump.
    for _ in range(3):  # Create each quatrain.
        # First line has no rhyming restrictions.
        output.append(make_line(hmm))

        # Second line has no rhyming restrictions.
        output.append(make_line(hmm))

        # Third line has to rhyme with first line.
        output.append(make_line(hmm, word_to_rhyme=w2i[output[-2].split(' ')[-1]]))

        # Fourth line has to rhyme with second line.
        output.append(make_line(hmm, word_to_rhyme=w2i[output[-2].split(' ')[-1]]))

    # Create the final couplet.
    output.append(make_line(hmm))
    output.append(make_line(hmm, word_to_rhyme=w2i[output[-1].split(' ')[-1]]))

    # Present the masterpiece.
    for line in output:
        print(line)