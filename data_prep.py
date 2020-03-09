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
        
        # Remove any whitespace from all of the line.
        sonnets = [[l.strip() for l in s if len(l) > 10] for s in sonnets]
        
        # Split all lines into list of words
        sonnets = [[re.findall('[a-z]+', l.lower()) for l in s] for s in sonnets]
        
    return sonnets