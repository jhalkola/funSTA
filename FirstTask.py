import numpy as np
import csv

'''
reads all values from groundtruth.csv and inputs them into a array
'''
def readStimulus():
    initial = True
    with open("groundtruth.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        items = []
        for row in reader:
            for value in row:
                items.append(float(value))
            row = np.array([row]).astype(np.float)
            if initial:
                groundtruth = row
                initial = False
            else:
                groundtruth = np.concatenate((groundtruth, row))
    sigma = np.transpose(np.array([items]))        # [s1 s2 s3 s1 s2 s3...]
    return sigma, groundtruth

'''
reads all measurements and makes the matrix M for least squares minimization
'''
def readMeasurements():
    with open("measurements.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        value_count = 0
        initial = True
        for row in reader:
            insert1 = np.zeros((1,12))[0]
            insert2 = np.zeros((1,12))[0]
            insert3 = np.zeros((1,12))[0]
            
            for value in row:
                value = float(value)
                insert1[value_count] = value
                insert1[3] = 1.0
            
                insert2[value_count + 4] = value
                insert2[7] = 1.0

                insert3[value_count + 8] = value
                insert3[11] = 1.0

                value_count += 1
            
            value_count = 0
            if initial == True:
                m = np.array([insert1, insert2, insert3])
                initial = False
            else:
                m = np.concatenate((m,  [insert1, insert2, insert3]))
    return m

    
def leastSquaresOptimatization():
    # sigma = M * beta
    # solved by: betaHat = ((transpose(M) * M)^-1) * transpose(M) * sigma
    trans_M = np.transpose(m)
    pseudo_inverse_M = np.linalg.inv(trans_M.dot(m)).dot(trans_M)       # ((transpose(M) * M)^-1) * transpose(M)
    beta_hat = pseudo_inverse_M.dot(sigma)
    return beta_hat

'''
get values out of betaHat for matrix A and vector B
'''
def getValuesFromBetaHat():
    a1 = []
    a2 = []
    a3 = []
    for i in range(0,11):
        if i < 3:
            a1.append(beta_hat[i][0])
        elif 3 < i < 7:
            a2.append(beta_hat[i][0])
        elif 7 < i < 11:
            a3.append(beta_hat[i][0])
    a = np.array([a1, a2, a3])      # A 3x3 matrix

    b1 = beta_hat[3][0]
    b2 = beta_hat[7][0]
    b3 = beta_hat[11][0]
    b = np.array([[b1, b2, b3]])
    return a, b

def correctionFunction(r):
    r = r.astype(np.float)
    y = a.dot(r) + b    # y = A*r + b
    return y

'''
calculates new values for the measurements and puts them in to a array
'''
def calculateNewResults():
    initial = True
    with open("measurements.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            new_row = correctionFunction(np.array(row))
            if initial:
                new_measurements = new_row
                initial = False
            else:
                new_measurements = np.concatenate((new_measurements, new_row))
    return new_measurements

'''
calculate sum-of-squares error from the new corrected measurements and groundtruth
'''
def calculateError():
    sum = 0
    for i in range(len(groundtruth) - 1):
        diff = groundtruth[i] - new_measurements[i]
        sqrd_diff = np.linalg.norm(diff) ** 2       # take norm of the error vector and square it
        sum = sum + sqrd_diff                       # sum-of-squres error
    return sum


if __name__ == "__main__":
    sigma, groundtruth = readStimulus()
    m = readMeasurements()
    beta_hat = leastSquaresOptimatization()
    a, b = getValuesFromBetaHat()

    print("3x3 matrix A is\n\n{}\n\n3x1 vector b is\n\n{}".format(a, b))

    new_measurements = calculateNewResults()
    sum_of_squares_error = calculateError()

    print("\nSum-of-squares error = {}\n".format(sum_of_squares_error))
