##############
# Name: Sehej Oberoi
# email: oberois@purdue.edu
# Date: 11/8/2020

import numpy as np
import sys
import os
import math
import random


def probAtt(att, outcome, totalTrue, totalFalse):
  countTrue = [1,1]
  countFalse = [1,1]
  split = 0
  groups = 2

  if att.max() == 1 and att.min() == 0:
    split = 0.5
    for i in range(0, len(att)):
      if outcome.iloc[i] == 1:
        countTrue[int(att.iloc[i])] += 1
      else:
        countFalse[int(att.iloc[i])] += 1
  else:
    if att.max() - att.min() <= 10:
      split = list(range(int(att.min()), int(att.max()) + 2))
    else:
      split = att.quantile([0.2, 0.4, 0.6, 0.8, 1]).tolist()

    countTrue = [1]*len(split)
    countFalse = [1]*len(split)
    for i in range(0, len(att)):
      for j in range(0, len(split)):
        if att.iloc[i] <= split[j] or j == len(split) - 1:
          if outcome.iloc[i] == 1:
            countTrue[j] += 1
          else:
            countFalse[j] += 1
          break
    groups = len(split)

  for i in range(0, len(countTrue)):
    countTrue[i] /= (1.0 * (totalTrue + groups))
    countFalse[i] /= (1.0 * (totalFalse + groups))

  return (countTrue, countFalse, split)




def predict(totalTrue, totalFalse, trueCounts, falseCounts, splits, row):
  trueProb = totalTrue/(1.0*(totalTrue + totalFalse))
  falseProb = totalFalse/(1.0*(totalTrue + totalFalse))
  denomT = trueProb
  denomF = falseProb

  for i in range(0, len(row)):
    if isinstance(splits[i], float):
      v = int(row[i] > 0)
      trueProb *= trueCounts[i][v]
      falseProb *= falseCounts[i][v]
      denomT *= trueCounts[i][v]
      denomF *= falseCounts[i][v]
    else:
      segment = 0
      for j in range(0, len(splits[i])):
        if row[i] <= int(splits[i][j]):
          trueProb *= trueCounts[i][j]
          falseProb *= falseCounts[i][j]
          denomT *= trueCounts[i][j]
          denomF *= falseCounts[i][j]
          break

  if trueProb > falseProb:
    return (1, trueProb/(denomF + denomT))
  else:
    return (0, falseProb/(denomF + denomT))

if __name__ == "__main__":
    # parse arguments
    import argparse
    import pandas as pd

    testDataFile = ''
    testLabelFile = ''
    trainDataFile = ''
    trainLabelFile = ''

    for arg in sys.argv[1:]:
      if 'test' in arg:
        if '.label' in arg:
          testLabelFile = arg
        elif '.data' in arg:
          testDataFile = arg
      if 'train' in arg:
        if '.label' in arg:
          trainLabelFile = arg
        elif '.data' in arg:
          trainDataFile = arg

    if testDataFile == '' or testLabelFile == '' or trainDataFile == '' or trainLabelFile == '':
      print("Not all arguments provided")
      exit

    trainData = pd.read_csv(trainDataFile, delimiter=',', index_col=None, engine='python')
    trainLabel = pd.read_csv(trainLabelFile, delimiter=',', index_col=None, engine='python')

    trainData = trainData.join(trainLabel)
    trainData[trainData.columns] = trainData[trainData.columns].apply(pd.to_numeric, errors='coerce')
    trainData = trainData.fillna(trainData.median())


    totalTrue = trainData[trainData.columns[-1]].sum()
    totalFalse = len(trainData) - totalTrue


    trueCounts = list()
    falseCounts = list()
    splits = list()

    for att in trainData.iloc[:,:-1]:
      counts = probAtt(trainData[att], trainData[trainData.columns[-1]], totalTrue, totalFalse)
      trueCounts.append(counts[0])
      falseCounts.append(counts[1])
      splits.append(counts[2])

    testData = pd.read_csv(testDataFile, delimiter=',', index_col=None, engine='python')
    testLabel = pd.read_csv(testLabelFile, delimiter=',', index_col=None, engine='python')


    correct = 0
    zeroOneLoss = 0
    squaredLoss = 0
    for i in range(0, len(testData)):
      prediction = predict(totalTrue, totalFalse, trueCounts, falseCounts, splits, testData.iloc[i].tolist())

      if int(prediction[0]) == testLabel.iloc[i].tolist()[0]:
        correct += 1
      else:
        zeroOneLoss += 1
        squaredLoss += (1.0 - prediction[1])*(1.0 - prediction[1])

    zeroOneLoss /= (1.0 * len(testData))
    squaredLoss /= (1.0 * len(testData))
    accuracy = correct / (1.0 * len(testData))

    print("ZERO-ONE LOSS=%.4f" % (zeroOneLoss))
    print("SQUARED LOSS=%.4f Test Accuracy=%.4f" % (squaredLoss, accuracy))
