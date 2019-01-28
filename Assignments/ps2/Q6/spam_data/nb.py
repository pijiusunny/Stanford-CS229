import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    state = {}
    N = matrix.shape[1]
    ###################

    # Compute the Bernoulli (prior) distribution for y, and store them in state
    m = matrix.shape[0]
    state['phi_1'] = sum(category == 1) / m
    state['phi_0'] = 1 - state['phi_1']
    
    # Compute the multinomial probabilities conditional on y=0 and y=1 in several steps
    # First, add the counts according to class labels
    probVector1 = np.sum(matrix[category == 1, :], axis = 0)
    probVector0 = np.sum(matrix[category == 0, :], axis = 0)
    
    # Second, perform Laplace smoothing
    probVector1 += 1
    probVector0 += 1
    
    # Finally, normalize the vectors to probabilities (each sums to one), and store them in state
    state['phi|y=1'] = probVector1 / sum(probVector1)
    state['phi|y=0'] = probVector0 / sum(probVector0)

    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################

    # For each example, compute two log-likelihood scores: logp(x, y=1) and logp(x, y=0)
    logLikelihood1 = np.matmul(matrix, np.log(state['phi|y=1'])) + np.log(state['phi_1'])
    logLikelihood0 = np.matmul(matrix, np.log(state['phi|y=0'])) + np.log(state['phi_0'])
    
    # For each example, compare the two log-likelihood scores and make prediction
    output[logLikelihood1 > logLikelihood0] = 1

    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print ('Error: %1.4f' % error)
    return error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
