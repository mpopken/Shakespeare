import numpy as np
from data_prep import parse_syllable_list

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = np.ones(self.L) / self.L

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        # First column
        for row in range(self.L):
            probs[0][row] = self.A_start[row]

        # All other columns
        for col in range(M):
            # calculate the probabilities of the column
            # using the probabilities from the last column
            for row in range(self.L):
                # maximize last row * transition matrix * observation matrix
                max_val = 0
                max_k = -1
                for k in range(self.L):
                    val = probs[col][k] * self.A[k][row] * self.O[k][x[col]]
                    if val > max_val:
                        max_val = val
                        max_k = k
                probs[col + 1][row] = max_val

                seqs[col + 1][row] = seqs[col][max_k] + str(max_k)
                if col + 1 == M:
                    final_val = 0
                    final_ind = -1
                    # Calculate the last column
                    for k in range(self.L):
                        val = probs[col][k] * self.A[k][row] * self.O[k][x[col]]
                        if val > final_val:
                            final_val = val
                            final_ind = k
        print('A: ', self.A)
        print('O: ', self.O)
        return seqs[M][final_ind]

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        # Via slide 73 of Lecture 13

        # Create the first two
        for z in range(self.L):
            alphas[0][z] = self.A_start[z]
            alphas[1][z] = self.O[z][x[0]] * self.A_start[z]

        col_sums = []
        # Fill in the rest
        for i in range(2, M + 1):
            for j in range(self.L):
                s = sum([alphas[i-1][k] * self.A[k][j] for k in range(self.L)])
                alphas[i][j] = self.O[j][x[i-1]] * s
            if normalize:
                denom = sum(alphas[i])
                for z in range(self.L):
                    alphas[i][z] = alphas[i][z] / denom

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for z in range(self.L):
            betas[M][z] = 1

        for i in range(M - 1, 0, -1):
            for j in range(self.L):
                arr = [betas[i+1][k] * self.A[j][k] * self.O[k][x[i]] for k in range(self.L)]
                betas[i][j] = sum(arr)
            if normalize:
                denom = sum(betas[i])
                for z in range(self.L):
                    betas[i][z] = betas[i][z] / denom

        return betas

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for _ in range(N_iters):
            A_num = np.zeros((self.L, self.L))
            A_denom = np.zeros((self.L, 1))
            
            O_num = np.zeros((self.L, self.D))
            O_denom = np.zeros((self.L, 1))

            for x in X:
                M = len(x)
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)

                # Forward-Backward algorithm
                for i in range(1, M+1):
                    prods = np.zeros((self.L, 1))
                    s = sum(alphas[i][j] * betas[i][j] for j in range(self.L))
                    for j in range(self.L):
                        prods[j] = alphas[i][j] * betas[i][j] / s
                    
                    for k in range(self.L):
                        if i < M:
                            A_denom[k] += prods[k]
                        O_num[k][x[i-1]] += prods[k]
                        O_denom[k] += prods[k]


                # Now want to calculate each marginal probability
                for i in range(1, M):
                    # denom = [[0 for _ in range(self.L)] for _ in range(self.L)]
                    denom = np.zeros((self.L, self.L))

                    for a in range(self.L):
                        for b in range(self.L):
                            denom[a][b] = alphas[i][a] * self.A[a][b] *  self.O[b][x[i]] * betas[i+1][b]

                    tot = 0
                    for j in range(self.L):
                        tot += sum(denom[j])
                    denom /= tot

                    for i in range(self.L):
                        for j in range(self.L):
                            A_num[i][j] += denom[i][j]

            self.A = A_num / A_denom
            self.O = O_num / O_denom

    def generate_emission(self):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        # Get the syllable list to make sure all lines are 10 syllables long.
        syls, _, i2w = parse_syllable_list('Syllable_dictionary.txt')
        syllable_count = 0

        emission = []
        states = []

        # Get the initial state and emission
        states.append(np.random.choice(self.L, p=self.A_start))
        emission.append(np.random.choice(self.D, p=self.O[states[-1]]))
        syllable_count += syls[i2w[emission[-1]]][0]
        
        while True:
            # Make sure we never have more than 10 syllables.
            assert syllable_count < 10  

            # Add a state and a word to the line.
            states.append(np.random.choice(self.L, p=self.A[states[-1]]))
            emission.append(np.random.choice(self.D, p=self.O[states[-1]]))
            prev_syl_count = syllable_count
            syllable_count += syls[i2w[emission[-1]]][0]

            # Check the end cases to make sure there are ten syllables in
            # every line. 
            if syllable_count > 10:
                # The last word may have a special end syllable count to still
                # make it valid.
                if syls[i2w[emission[-1]]][2] is not None:
                    end_count = syls[i2w[emission[-1]]][2]
                    proper_count = syls[i2w[emission[-1]]][0]
                    syllable_count += end_count - proper_count
                
                # If it became valid, break and return; otherwise get rid of
                # the last word and try again.
                if syllable_count != 10:
                    states = states[:-1]
                    emission = emission[:-1]
                    syllable_count = prev_syl_count
                    continue
                else:
                    break
            
            if syllable_count == 10:
                # If we have a proper syllable count and no funny stuff with
                # the end word syllable counts, break and return.
                if syls[i2w[emission[-1]]][2] is None:
                    break
                # If there is some funny stuff, get rid of the last word and
                # try again.
                else:
                    states = states[:-1]
                    emission = emission[:-1]
                    syllable_count = prev_syl_count
                    continue

        return emission, states

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        n_words:    Number of words to choose from.
        
        N_iters:    The number of iterations to train on.
    '''
    
    # Compute L and D.
    L = n_states
    D = max([max(line) for line in X]) + 1

    # Randomly initialize and normalize matrix A.
    A = np.random.rand(L, L)
    A /= A.sum(axis=1).reshape((-1, 1))
    
    # Randomly initialize and normalize matrix O.
    O = np.random.rand(L, D)
    O /= O.sum(axis=1).reshape((-1, 1))

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
