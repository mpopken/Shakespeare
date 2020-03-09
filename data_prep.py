import re

def get_sonnets_from_file(path):
    '''
    Convert a text file to a 2d array of sonnets, where index (i, j, k)
    is the kth word in the jth line of the ith sonnet.
    
    args:
        path. The file path to get the sonnets from.
    returns:
        3d array of sonnets.
    '''
    
    with open(path, 'r') as f:
        # Read all of the text from the file.
        full_text = f.read()
        
    # Split the text into sonnets
    sonnets = full_text.split('\n\n')
    sonnets = [s.strip().split('\n') for s in sonnets if len(s) > 10]
    
    # Remove any whitespace from all of the lines
    sonnets = [[l.strip() for l in s if len(l) > 10] for s in sonnets]
    
    # Split all lines into list of words    
    sonnets = [[re.findall("[a-z'-]+", l.lower()) for l in s] for s in sonnets]
        
    return sonnets

def parse_syllable_list(path):
    '''
    Get a dictionary mapping words to syllable counts from the file.

    args:
        path. Path to the file containing syllable counts.
    returns:
        A dictionary mapping words to syllable count.
          Structure of dictionary is {'word': (c1, c2, c2)}.

          If the word is always some number of syllables, then c1 is that
            number, c2 and c3 are None.
          If the word can be multiple numbers of syllables, then c1 and c2 are
            these counts, and c3 is None.
          If the word has a different number of syllables if it is at the end
            of a line, then c1 is the typical number of syllables, c2 is None,
            and c3 is the number of syllables if at the end of the line.
        
          Note: most of the time c2 and c3 are None.
    '''

    with open(path, 'r') as f:
        # Read the syllable list from the path.
        full_text = f.read()

    smap = {}
    w2i = {}
    i2w = {}
    
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
            smap[segs[0]] = (int(segs[1]), None, None)

        # If there are three items in a line, then the word can be multiple
        # syllable lengths.
        elif len(segs) == 3:
            if 'E' in segs[1]:
                smap[segs[0]] = (int(segs[2]), None, int(segs[1][1:]))
            elif 'E' in segs[2]:
                smap[segs[0]] = (int(segs[1]), None, int(segs[2][1:]))
            else:
                smap[segs[0]] = (int(segs[1]), int(segs[2]), None)
    
    return smap, w2i, i2w