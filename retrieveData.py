import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import math


def graphData(x):
  plt.plot(x)
  plt.show()

def dayNumberSineComponent(n):
  x=2*math.pi*n
  x=x/365.25
  x=math.sin(x)
  return x
def dayNumberCosineComponent(n):
  x=2*math.pi*n
  x=x/365.25
  x=math.cos(x)
  return x

def hourNumberSineComponent(n):
  x=2*math.pi*n
  x=x/24
  x=math.sin(x)
  return x
def hourNumberCosineComponent(n):
  x=2*math.pi*n
  x=x/24
  x=math.cos(x)
  return x


def getInputs(printGraph=False):
    """
      first function to return Inputs for Neural Network

      Parameters:
        printGraph: (default:False) will print graph of each input parameter

      Returns:
      a list of following parameters
       [train_X,train_Y,dev_X,dev_Y,test_X,test_Y]
       with ratios [0.75,0.125,0.125]
        X having 3 columns with
          column1=Ap index values
          column2=F10.7 values
          column3=gps values
        and n rows where n= no of examples

     """
    fp = open('Ap.txt', 'r')
    Ap = fp.read().split()
    Ap = [float(i) for i in Ap]
    fp.close()

    fp = open('F10.7.txt', 'r')
    F10 = fp.read().split()
    F10 = [float(i) for i in F10]
    fp.close()

    fp = open('gpsdata.txt', 'r')
    gpsData = fp.read().split()
    gpsData = [float(i) for i in gpsData]
    fp.close()

    if (printGraph == True):
        graphData(Ap)
        graphData(F10)
        graphData(gpsData)

    size = 70080
    train = 0.75
    dev = 0.125
    test = 0.125

    minimum = 0;
    maximum = int(size * train)
    train_Ap = np.asarray(Ap[minimum:maximum]).reshape(maximum, 1)
    train_F10 = np.asarray(F10[minimum:maximum]).reshape(maximum, 1)
    train_gpsData = np.asarray(gpsData[minimum:maximum]).reshape(maximum, 1)

    train_X = np.array([train_Ap, train_F10, train_gpsData]).T.reshape(maximum, 3)

    train_Y = np.asarray(gpsData[minimum + 1:maximum + 1]).reshape(maximum, 1)

    minimum = int(size * train);
    maximum = int(size * train) + int(size * dev);
    dev_size = int(size * dev)
    dev_Ap = np.asarray(Ap[minimum:maximum]).reshape(dev_size, 1)
    dev_F10 = np.asarray(F10[minimum:maximum]).reshape(dev_size, 1)
    dev_gpsData = np.asarray(gpsData[minimum:maximum]).reshape(dev_size, 1)

    dev_X = np.array([dev_Ap, dev_F10, dev_gpsData]).T.reshape(dev_size, 3)

    dev_Y = np.asarray(gpsData[minimum + 1:maximum + 1]).reshape(dev_size, 1)

    minimum = maximum
    maximum = maximum + int(size * test)
    test_size = int(size * test) - 1
    test_Ap = np.asarray(Ap[minimum:maximum - 1]).reshape(test_size, 1)
    test_F10 = np.asarray(F10[minimum:maximum - 1]).reshape(test_size, 1)
    test_gpsData = np.asarray(gpsData[minimum:maximum - 1]).reshape(test_size, 1)

    test_X = np.array([test_Ap, test_F10, test_gpsData]).T.reshape(test_size, 3)

    test_Y = np.asarray(gpsData[minimum + 1:maximum]).reshape(test_size, 1)

    return [train_X, train_Y, dev_X, dev_Y, test_X, test_Y]


def getInputs2(printGraph=False):
    """
      first function to return Inputs for Neural Network

      Parameters:
        printGraph: (default:False) will print graph of each input parameter

      Returns:
      a list of following parameters
       [train_X,train_Y,dev_X,dev_Y,test_X,test_Y]
       with ratios [0.75,0.125,0.125]
        X having 3 columns with
          column1=Ap index values
          column2=F10.7 values
          column3=gps values
        and n rows where n= no of examples

     """
    fp = open('Ap.txt', 'r')
    Ap = fp.read().split()
    Ap = [float(i) for i in Ap]
    fp.close()

    fp = open('F10.7.txt', 'r')
    F10 = fp.read().split()
    F10 = [float(i) for i in F10]
    fp.close()

    fp = open('gpsdata.txt', 'r')
    gpsData = fp.read().split()
    gpsData = [float(i) for i in gpsData]
    fp.close()

    DN = [];
    HR = [];
    dv = 1;
    hv = 1;
    DNS = [];
    DNC = [];
    HRS = [];
    HRC = [];
    for i in range(0, 70080):
        HR.append(hv)
        HRS.append(hourNumberSineComponent(hv))
        HRC.append(hourNumberCosineComponent(hv))
        DN.append(dv)
        DNS.append(dayNumberSineComponent(dv))
        DNC.append(dayNumberCosineComponent(dv))

        hv += 1
        if (hv == 25):
            dv += 1;
            hv = 1;
        if (dv == 366):
            dv = 1;
    #   for i in range(len(DN)):
    #     print(DN[i],HR[i],DNS[i],DNC[i],HRS[i],HRC[i])


    if (printGraph == True):
        graphData(DNS)
        graphData(DNC)
        graphData(HRS)
        graphData(HRC)
        graphData(Ap)
        graphData(F10)
        graphData(gpsData)

    size = 70080
    train = 0.75
    dev = 0.125
    test = 0.125

    minimum = 0;
    maximum = int(size * train)
    train_Ap = np.asarray(Ap[minimum:maximum]).reshape(maximum, 1)
    train_F10 = np.asarray(F10[minimum:maximum]).reshape(maximum, 1)
    train_gpsData = np.asarray(gpsData[minimum:maximum]).reshape(maximum, 1)
    train_DNS = np.asarray(DNS[minimum:maximum]).reshape(maximum, 1)
    train_DNC = np.asarray(DNC[minimum:maximum]).reshape(maximum, 1)
    train_HRS = np.asarray(HRS[minimum:maximum]).reshape(maximum, 1)
    train_HRC = np.asarray(HRC[minimum:maximum]).reshape(maximum, 1)

    train_X = np.array([train_Ap, train_F10, train_gpsData, train_DNS, train_DNC, train_HRS, train_HRC]).T.reshape(
        maximum, 7)

    train_Y = np.asarray(gpsData[minimum + 1:maximum + 1]).reshape(maximum, 1)

    minimum = int(size * train);
    maximum = int(size * train) + int(size * dev);
    dev_size = int(size * dev)
    dev_Ap = np.asarray(Ap[minimum:maximum]).reshape(dev_size, 1)
    dev_F10 = np.asarray(F10[minimum:maximum]).reshape(dev_size, 1)
    dev_gpsData = np.asarray(gpsData[minimum:maximum]).reshape(dev_size, 1)
    dev_DNS = np.asarray(DNS[minimum:maximum]).reshape(dev_size, 1)
    dev_DNC = np.asarray(DNC[minimum:maximum]).reshape(dev_size, 1)
    dev_HRS = np.asarray(HRS[minimum:maximum]).reshape(dev_size, 1)
    dev_HRC = np.asarray(HRC[minimum:maximum]).reshape(dev_size, 1)

    dev_X = np.array([dev_Ap, dev_F10, dev_gpsData, dev_DNS, dev_DNC, dev_HRS, dev_HRC]).T.reshape(dev_size, 7)

    dev_Y = np.asarray(gpsData[minimum + 1:maximum + 1]).reshape(dev_size, 1)

    minimum = maximum
    maximum = maximum + int(size * test)
    test_size = int(size * test) - 1
    test_Ap = np.asarray(Ap[minimum:maximum - 1]).reshape(test_size, 1)
    test_F10 = np.asarray(F10[minimum:maximum - 1]).reshape(test_size, 1)
    test_gpsData = np.asarray(gpsData[minimum:maximum - 1]).reshape(test_size, 1)
    test_DNS = np.asarray(DNS[minimum:maximum - 1]).reshape(test_size, 1)
    test_DNC = np.asarray(DNC[minimum:maximum - 1]).reshape(test_size, 1)
    test_HRS = np.asarray(HRS[minimum:maximum - 1]).reshape(test_size, 1)
    test_HRC = np.asarray(HRC[minimum:maximum - 1]).reshape(test_size, 1)

    test_X = np.array([test_Ap, test_F10, test_gpsData, test_DNS, test_DNC, test_HRS, test_HRC]).T.reshape(test_size, 7)

    test_Y = np.asarray(gpsData[minimum + 1:maximum]).reshape(test_size, 1)

    return [train_X, train_Y, dev_X, dev_Y, test_X, test_Y]

train_X,train_Y,dev_X,dev_Y,test_X,test_Y=getInputs2()