from numpy.core.numeric import Infinity
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math

def Sort(sub_li):
  
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of 
    # sublist lambda has been used

    sub_li.sort(key = lambda x: x[1])
    return sub_li

def getDistance(p1,p2,p):

    diff = p1-p2
    diff = abs(diff)
    if p == np.inf:
        return np.max(diff)    
    power = np.power(diff,p)
    sum = np.sum(power)
    return np.power(sum,(1/p))

def Label_by_k_closest_points(pointToLabel,S_points,S_labels,k,p):
    # find the closest k points
    min_distances = [(0,np.inf)] * k
    for i in range(len(S_points)):
        dis = getDistance(pointToLabel,S_points[i],p)
        cell = [i,dis]
        if (dis < min_distances[k-1][1]):
            min_distances[k-1] = cell
            min_distances = Sort(min_distances)
    min_distances = np.array(min_distances)
    # extract their indexes
    indexes = min_distances[:,0]
    indexes = indexes.astype(int)
    sum = S_labels[indexes]
    sum = np.array(sum)
    sum = np.sum(sum)
    if (sum > 0):
        return 1
    else:
        return -1

def run():

    # create a df of points (x,y) and their labels
    df = pd.read_table('rectangle.txt', delim_whitespace=True, names=('x', 'y', 'label'))
    points_df = df[['x', 'y']].copy()
    labels_df = df[['label']].copy()
    points_df = np.array(points_df)
    labels_df = np.array(labels_df)
    # Split the data randomly into 1/2 test (T) and 1/2 train (S)
    S_points, T_points, S_labels, T_labels = train_test_split(points_df, labels_df, test_size=0.5, shuffle=True)
    TrainErrors = []
    TestErrors = []
    for k in range(1,10,2):
        for p in (1,2,np.inf):

            Labels = []
            for i in range(len(S_points)):    
                Label_of_point_i = Label_by_k_closest_points(S_points[i],S_points,S_labels,k,p)
                Labels.append(Label_of_point_i)
            #find the error rate
            Labels = np.array(Labels)
            Labels = Labels.reshape(75,1)
            b = np.array(S_labels - Labels)
          
            b = b**2
            b = b/4
            a = np.sum(b)
            TrainErrors.append(a/75)
            Labels = []
            for i in range(len(T_points)):    
                Label_of_point_i = Label_by_k_closest_points(T_points[i],S_points,S_labels,k,p)
                Labels.append(Label_of_point_i)
             #find the error rate
            Labels = np.array(Labels)
            Labels = Labels.reshape(75,1)
            b = np.array(T_labels - Labels)
           
            b = b**2
            b = b/4
            a = np.sum(b)
            TestErrors.append(a/75)
    return np.array(TrainErrors),np.array(TestErrors)



def main():

    TrainErrors = np.zeros(15)
    TestErrors = np.zeros(15)
    TrainErrors,TestErrors = run()
    print("k = 1, p = 1")
    print("train error = ",TrainErrors[0])
    print("test error = ",TestErrors[0])
    print("k = 1, p = 2")
    print("train error = ",TrainErrors[1])
    print("test error = ",TestErrors[1])
    print("k = 1, p = inf")
    print("train error = ",TrainErrors[2])
    print("test error = ",TestErrors[2])
    print("k = 3, p = 1")
    print("train error = ",TrainErrors[3])
    print("test error = ",TestErrors[3])
    print("k = 3, p = 2")
    print("train error = ",TrainErrors[4])
    print("test error = ",TestErrors[4])
    print("k = 3, p = inf")
    print("train error = ",TrainErrors[5])
    print("test error = ",TestErrors[5])
    print("k = 5, p = 1")
    print("train error = ",TrainErrors[6])
    print("test error = ",TestErrors[6])
    print("k = 5, p = 2")
    print("train error = ",TrainErrors[7])
    print("test error = ",TestErrors[7])
    print("k = 5, p = inf")
    print("train error = ",TrainErrors[8])
    print("test error = ",TestErrors[8])
    print("k = 7, p = 1")
    print("train error = ",TrainErrors[9])
    print("test error = ",TestErrors[9])
    print("k = 7, p = 2")
    print("train error = ",TrainErrors[10])
    print("test error = ",TestErrors[10])
    print("k = 7, p = inf")
    print("train error = ",TrainErrors[11])
    print("test error = ",TestErrors[11])
    print("k = 9, p = 1")
    print("train error = ",TrainErrors[12])
    print("test error = ",TestErrors[12])
    print("k = 9, p = 2")
    print("train error = ",TrainErrors[13])
    print("test error = ",TestErrors[13])
    print("k = 9, p = inf")
    print("train error = ",TrainErrors[14])
    print("test error = ",TestErrors[14])

    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    print("#########################################################")

    for i in range(100):
        tr, tst =  run()
        TrainErrors = TrainErrors +tr
        TestErrors = TestErrors +tst

    TrainErrors = TrainErrors/100
    TestErrors = TestErrors/100
    print("k = 1, p = 1")
    print("train error = ",TrainErrors[0])
    print("test error = ",TestErrors[0])
    print("k = 1, p = 2")
    print("train error = ",TrainErrors[1])
    print("test error = ",TestErrors[1])
    print("k = 1, p = inf")
    print("train error = ",TrainErrors[2])
    print("test error = ",TestErrors[2])
    print("k = 3, p = 1")
    print("train error = ",TrainErrors[3])
    print("test error = ",TestErrors[3])
    print("k = 3, p = 2")
    print("train error = ",TrainErrors[4])
    print("test error = ",TestErrors[4])
    print("k = 3, p = inf")
    print("train error = ",TrainErrors[5])
    print("test error = ",TestErrors[5])
    print("k = 5, p = 1")
    print("train error = ",TrainErrors[6])
    print("test error = ",TestErrors[6])
    print("k = 5, p = 2")
    print("train error = ",TrainErrors[7])
    print("test error = ",TestErrors[7])
    print("k = 5, p = inf")
    print("train error = ",TrainErrors[8])
    print("test error = ",TestErrors[8])
    print("k = 7, p = 1")
    print("train error = ",TrainErrors[9])
    print("test error = ",TestErrors[9])
    print("k = 7, p = 2")
    print("train error = ",TrainErrors[10])
    print("test error = ",TestErrors[10])
    print("k = 7, p = inf")
    print("train error = ",TrainErrors[11])
    print("test error = ",TestErrors[11])
    print("k = 9, p = 1")
    print("train error = ",TrainErrors[12])
    print("test error = ",TestErrors[12])
    print("k = 9, p = 2")
    print("train error = ",TrainErrors[13])
    print("test error = ",TestErrors[13])
    print("k = 9, p = inf")
    print("train error = ",TrainErrors[14])
    print("test error = ",TestErrors[14])

    my_xticks = ("1,1", "1,2","1,inf","3,1","3,2","3,inf","5,1","5,2","5,inf",
    "7,1","7,2","7,inf","9,1","9,2","9,inf")
    x= np.arange(0,15)
    plt.xticks(x, my_xticks)
    plt.plot(x, TrainErrors/100,'b')
    plt.plot(x, TestErrors/100,'r')
    plt.title('K-nearset')
    plt.ylabel('Error')
    plt.xlabel('k nearst and p distance')
    plt.legend(['Train','Test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()