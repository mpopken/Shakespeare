'''
This file contains many of the utility functions and data preprocessing
functions that are necessary for processing the Shakespeare file and generating
sonnets using the Hidden Markov Model.
'''

import numpy as np
import re
import nltk

def get_sonnets_from_file(path):
    '''
    Convert a text file to a 2d array of sonnets, where index (i, j, k)
    is the kth word in the jth line of the ith sonnet.
    
    args:
        path. The file path to get the sonnets from.
    returns:
        3d array of sonnets.
    '''
    
    # Make sure the input is a valid type.
    assert type(path) is str

    with open(path, 'r') as f:
        # Read all of the text from the file.
        full_text = f.read()
        
    # Split the text into sonnets
    sonnets = full_text.split('\n\n')
    sonnets = [s.strip().split('\n') for s in sonnets if len(s) > 10]
    
    # Remove any whitespace from all of the lines
    sonnets = [[l.strip() for l in s if len(l) > 10] for s in sonnets]
    
    # Split all lines into list of words    
    sonnets = [[re.findall("[a-z'-.?]+", l.lower()) for l in s] for s in sonnets]
        
    return sonnets



def parse_syllable_list(path):
    '''
    Get a dictionary mapping words to syllable counts from the file.

    args:
        path. Path to the file containing syllable counts.
    returns:
        A dictionary mapping words to syllable count,
          Structure of dictionary is {'word': (c1, c2, c2)}.
            If the word is always some number of syllables, then c1 is that
              number, c2 is None.
            If the word has a different number of syllables if it is at the end
              of a line, then c1 is the typical number of syllables and c2 is
              the number of syllables if at the end.
        A dictionary mapping each word to its index,
        A dictionary mapping each index to its word.
    '''

    # Make sure the input is a valid type.
    assert type(path) is str

    with open(path, 'r') as f:
        # Read the syllable list from the path.
        full_text = f.read()

    smap = {}
    w2i = {}
    i2w = {}
    w2i[None] = None
    i2w[None] = None

    # Split the text into lines
    for i, line in enumerate(full_text.split('\n')):
        segs = line.split()

        # Assign an index to every word.
        if len(segs) > 0:
            w2i[segs[0]] = i
            i2w[i] = segs[0]

        # If there are two items in a line, the first is the words and the
        # second is the number of syllables.
        if len(segs) == 2:
            smap[segs[0]] = (int(segs[1]), None)

        # If there are three items in a line, then the word can be multiple
        # syllable lengths.
        elif len(segs) == 3:
            if 'E' in segs[1]:
                smap[segs[0]] = (int(segs[2]), int(segs[1][1:]))
            elif 'E' in segs[2]:
                smap[segs[0]] = (int(segs[1]), int(segs[2][1:]))
            else:
                smap[segs[0]] = (int(segs[2]), None)
    
    return smap, w2i, i2w



def rhyme(word, word_list):
    '''
    Get a list of words that rhyme with a given word.

    args:
        word. The word to rhyme with.
        word_list. The list of words to choose rhymes from.
    returns:
        A list of all of the word in word_list that rhyme with word.
    '''

    # Make sure the input is a valid type.
    assert word is None or type(word) is str
    # assert type(word_list) is list  and \
    #        len(word_list) > 0       and \
    #        type(word_list[0]) is str

    # Special case: if we get special argument None then we return None back.
    if word is None:
        return None

    # Dictionary that maps words to their pronunciations.
    pron = nltk.corpus.cmudict.dict()
    if word not in pron.keys():
        # Return an empty list if nltk doesn't recognise the word.
        return []

    # Get the pronunciations of this word
    this = (word, pron[word])

    # Ugly way of getting rhymes using only a single list comprehension.
    rhymes = [w for w in word_list if w in pron.keys() and
                                      w != word        and
                                      pron[w][0][-2:] == this[1][0][-2:]]
                                      
    return rhymes

def get_sequences(sonnets, length=40):
    '''
    Gets all sequences of characters of a specified length, converted to ints.
    
    args:
        sonnets. A 3D array of sonnets where each sonnet is a 2D array
        length (optional). length of the sequences to get
    returns:
        All sequences of characters of the desired length,
        generated from every sonnet in sonnets. Each number represents
        a unique character.
        The character after each sequence
        All of the unique characters
    '''
    sequences = []
    outputs = []
    
    # Make it back into one string
    raw_string = ''
    for sonnet in sonnets:
        raw_string += '\n'.join([' '.join(sonnet[i]).lower() for i in range(len(sonnet))])
        raw_string += '\n'
    # A list of all unique characters
    charlst = sorted(list(set(raw_string)))
    # A dictionary mapping a character to its index in chars
    char_to_int = dict((c, i) for i,c in enumerate(charlst))
    #print(char_to_int)
    
    for sonnet in sonnets:
        # This combines every character in the sonnet into one string
        # of characters ... Sorry
        chars = '\n'.join([' '.join(sonnet[i]).lower() for i in range(len(sonnet))])

        # Get all subsequences of length = length
        for i in range(len(chars) - length - 1):
            input_sequence = chars[i:i+length]
            output = chars[i+length]
            sequences.append(seq_to_ints(input_sequence, char_to_int))
            outputs.append(seq_to_ints(output, char_to_int))
    
    return sequences, outputs, charlst

# Dict is a dictionary mapping chars to ints
def seq_to_ints(sequence, dic):
    return [dic[char] for char in sequence]

# Not needed
def split_sequences(sequences, ratio):
    '''
        Given sequences of characters, split them into training
        and validation sets.
        
        args:
            sequences. All of the character sequences
            ratio (optional). What % of thhe data to use for training
        returns:
            A training set and a validation set
    '''
    permutation = np.random.permutation(sequences)
    split_index = int(len(permutation) * ratio)
    
    return permutation[:split_index], permutation[split_index:]