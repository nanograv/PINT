import matplotlib.pyplot as plt
import numpy as np

def readFile(filename):
    '''
    Read a file with fixed number of colums 
    '''
    f = open(filename,'r')
    firstLine = f.readline()
#    firstLine - firstLine.strip('\n')
    firstLine = firstLine.split()
    numClum = len(firstLine)
    data = np.zeros((1,numClum))
    data[0] = [float(x) for x in firstLine]
    for line in f.readlines():
        line  = line.strip('\n')
        linesplt  = line.split()
        data = np.vstack((data,[float(x) for x in linesplt]))
    return data

def plotDiff(data1,data2,xaxis = []):
    '''
    Plot the difference between two row of data 
    '''
    diff = data1-data2
    if(xaxis == []):  
        plt.figure(1)
        plt.plot(diff)
        plt.ylabel("Difference between data1 and data2")
        plt.show()
    else:
        plt.figure(1)
        plt.plot(xaxis,diff)
        plt.ylabel("Difference")
        plt.show()    
