import numpy as np
import pandas as pd

data = pd.read_csv('trainingdata.csv')

concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])

def candidate_elimination(concepts, target):
    '''
    candidate_elimination() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a numpy array with all the features
        target - a numpy array with corresponding output values
    '''
    specific_h = concepts[0]
    print("\nInitialization of specific_h and general_h")
    print(specific_h)

    # Initialize general_h to have all '?' initially
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    print(general_h)

    for i, h in enumerate(concepts):
        # If the target is positive, update specific_h and general_h
        if target[i] == "Yes":
            for j in range(len(specific_h)):
                if h[j] != specific_h[j]:
                    specific_h[j] = '?'
                    general_h[j][j] = '?'
        # If the target is negative, update general_h
        else:
            for j in range(len(specific_h)):
                if h[j] != specific_h[j]:
                    general_h[j][j] = specific_h[j]
                else:
                    general_h[j][j] = '?'

        print("\nSteps of Candidate Elimination Algorithm", i+1)
        print(specific_h)
        print(general_h)

    # Remove rows with all '?' from general_h
    general_h = [row for row in general_h if row != ['?'] * len(row)]

    return specific_h, general_h

s_final, g_final = candidate_elimination(concepts, target)
print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")
