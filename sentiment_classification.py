'''
  Author: Julia Jeng, Shu Wang, Arman Anwar
  Brief: AIT 726 Homework 2
  Usage:
      Put file 'sentiment_classification.py' and folder 'twitter' in the same folder.
  Command to run:
      python sentiment_classification.py
  Description:
      Build and train a feed forward neural network (FFNN) with 2 layers with hidden vector size 20.
      Initalized weights: random weights.
      Loss function: mean squared error.
      Activation function: sigmoid.
      Learning rate: 0.01.
      Train/valid rate: 4:1
      Emoticon tokenizer: TweetTokenizer
'''

import os
import re
import sys
import math
import random
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from itertools import chain
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
nltk.download('stopwords')
nltk.download('punkt')

# global path
logPath = './sentiment_classification.txt'
datPath = './tweet/'
tmpPath = './tmp/'

# Logger: redirect the stream on screen and to file.
class Logger(object):
    def __init__(self, filename = "log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

# The main function.
def main():
    # initialize the log file.
    sys.stdout = Logger(logPath)

    print("-- AIT726 Homework 2 from Julia Jeng, Shu Wang, and Arman Anwar --")

    # create the vocabulary.
    if (os.path.exists(tmpPath + '/Train.npz') and
        os.path.exists(tmpPath + '/Test.npz') and
        os.path.exists(tmpPath + '/Vocab.npz')):
        print('Successfully find data from \'' + tmpPath + '\'.')
    else:
        CreateVocabulary()

    # run demo.
    DemoFFNN('Stem')
    DemoFFNN('noStem')
    return

# a demo of neural network classifier.
def DemoFFNN(lStem = 'noStem'):
    '''
    a demo of neural network classifier.
    :param lStem: stem setting - 'noStem', 'Stem'
    :return: none
    '''
    # input validation.
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return

    # extract training features with 'lStem' dataset.
    if os.path.exists(tmpPath + '/featTrain_' + lStem + '.npy'):
        featTrain = np.load(tmpPath + '/featTrain_' + lStem + '.npy')
        print('Successfully load ' + tmpPath + '/featTrain_' + lStem + '.npy')
    else:
        featTrain = ExtractFeatures('Train', lStem)
        np.save(tmpPath + '/featTrain_' + lStem + '.npy', featTrain)

    # get the model parameters.
    model = TrainFFNN(featTrain)

    # extract testing features with 'lStem' dataset.
    if os.path.exists(tmpPath + '/featTest_' + lStem + '.npy'):
        featTest = np.load(tmpPath + '/featTest_' + lStem + '.npy')
        print('Successfully load ' + tmpPath + '/featTest_' + lStem + '.npy')
    else:
        featTest = ExtractFeatures('Test', lStem)
        np.save(tmpPath + '/featTest_' + lStem + '.npy', featTest)

    # get testing predictions using model parameters.
    accuracy, confusion = TestFFNN(model, featTest)

    # output the results on screen and to files.
    OutputFFNN(accuracy, confusion, lStem)
    # debug
    return

# Read train/test sets and create vocabulary.
def CreateVocabulary():
    '''
    read train and test sets and create vocabulary.
    :return: none
    '''
    # pre-process the data.
    def Preprocess(data):
        # remove url
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        data = re.sub(pattern, '', data)
        # remove html special characters.
        pattern = r'&[(amp)(gt)(lt)]+;'
        data = re.sub(pattern, '', data)
        # remove independent numbers.
        pattern = r' \d+ '
        data = re.sub(pattern, ' ', data)
        # lower case capitalized words.
        pattern = r'([A-Z][a-z]+)'
        def LowerFunc(matched):
            return matched.group(1).lower()
        data = re.sub(pattern, LowerFunc, data)
        # remove hashtags.
        pattern = r'[@#]([A-Za-z]+)'
        data = re.sub(pattern, '', data)
        return data

    # remove stop words.
    def RemoveStop(data):
        dataList = data.split()
        for item in dataList:
            if item.lower() in stopwords.words('english'):
                dataList.remove(item)
        dataNew = " ".join(dataList)
        return dataNew

    # get tokens.
    def GetTokens(data):
        # use tweet tokenizer.
        tknzr = TweetTokenizer()
        tokens = tknzr.tokenize(data)
        tokensNew = []
        # tokenize at each punctuation.
        pattern = r'[A-Za-z]+\'[A-Za-z]+'
        for tk in tokens:
            if re.match(pattern, tk):
                subtokens = word_tokenize(tk)
                tokensNew = tokensNew + subtokens
            else:
                tokensNew.append(tk)
        return tokensNew

    # process tokens with stemming.
    def WithStem(tokens):
        porter = PorterStemmer()
        tokensStem = []
        for tk in tokens:
            tokensStem.append(porter.stem(tk))
        return tokensStem

    # if there is no 'tmp' folder, create one.
    if not os.path.exists(tmpPath):
        os.mkdir(tmpPath)
    if not os.path.exists(datPath):
        print('[ERROR]: Cannot find the data path \'' + datPath + '\'.')

    # read the training data.
    labelTrain = []
    dataTrain = []
    dataTrainStem = []
    for root, ds, fs in os.walk(datPath + '/train/'):
        for file in fs:
            fullname = os.path.join(root, file)
            # get the training label.
            if "positive" in fullname:
                label = 1
            else: # "negative" in fullname
                label = 0
            labelTrain.append(label)
            # get the training data.
            data = open(fullname, encoding="utf8").read()
            # print(data)
            # preprocess the data.
            data = Preprocess(data)
            # print(data)
            # remove stop words.
            data = RemoveStop(data)
            # print(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTrain.append(tokens)
            # print(tokens)
            # get the stemmed tokens for the data.
            tokensStem = WithStem(tokens)
            dataTrainStem.append(tokensStem)
            # print(tokensStem)
    print('Load TrainSet: %d/%d positive/negative samples.' % (sum(labelTrain), len(labelTrain)-sum(labelTrain)))
    np.savez(tmpPath + '/Train.npz', labelTrain = labelTrain, dataTrain = dataTrain, dataTrainStem = dataTrainStem)

    # build the vocabulary from training set.
    vocab = list(set(list(chain.from_iterable(dataTrain))))
    vocabStem = list(set(list(chain.from_iterable(dataTrainStem))))
    print('Vocabulary: %d items.' % len(vocab))
    print('Vocabulary (stem): %d items.' % len(vocabStem))
    np.savez(tmpPath + '/Vocab.npz', vocab = vocab, vocabStem = vocabStem)

    # read the testing data.
    labelTest = []
    dataTest = []
    dataTestStem = []
    for root, ds, fs in os.walk(datPath + '/test/'):
        for file in fs:
            fullname = os.path.join(root, file)
            # get the testing label.
            if "positive" in fullname:
                label = 1
            else: # "negative" in fullname
                label = 0
            labelTest.append(label)
            # get the testing data.
            data = open(fullname, encoding="utf8").read()
            # print(data)
            # preprocess the data.
            data = Preprocess(data)
            # print(data)
            # remove stop words.
            data = RemoveStop(data)
            # print(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTest.append(tokens)
            # print(tokens)
            # get the stemmed tokens for the data.
            tokensStem = WithStem(tokens)
            dataTestStem.append(tokensStem)
            # print(tokensStem)
    print('Load TestSet: %d/%d positive/negative samples.' % (sum(labelTest), len(labelTest)-sum(labelTest)))
    np.savez(tmpPath + '/Test.npz', labelTest = labelTest, dataTest = dataTest, dataTestStem = dataTestStem)
    return

# extract tfidf features for a 'dataset' with or without 'stem'
def ExtractFeatures(dataset = 'Train', lStem = 'noStem'):
    '''
    extract features for a 'dataset' with or without 'stem'
    :param dataset: dataset type - 'Train', 'Test'
    :param lStem: stem setting - 'noStem', 'Stem'
    :return: tfidf feature - D * V
    '''
    # input validation.
    if dataset not in ['Train', 'Test']:
        print('Error: dataset input invalid!')
        return
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return

    # sparse the corresponding dataset.
    dset = np.load(tmpPath + dataset + '.npz', allow_pickle = True)
    if 'Stem' == lStem:
        data = dset['data' + dataset + lStem]
    else:
        data = dset['data' + dataset]
    D = len(data)

    # sparse the corresponding vocabulary.
    vset = np.load(tmpPath + '/Vocab.npz', allow_pickle = True)
    if 'Stem' == lStem:
        vocab = vset['vocab' + lStem]
    else:
        vocab = vset['vocab']
    V = len(vocab)
    vocabDict = dict(zip(vocab, range(V)))

    # get the feature matrix (tfidf):
    # get freq and bin features.
    termFreq = np.zeros((D, V))
    termBin = np.zeros((D, V))
    for ind, doc in enumerate(data):
        for item in doc:
            if item in vocabDict:
                termFreq[ind][vocabDict[item]] += 1
                termBin[ind][vocabDict[item]] = 1

    # get tf (1+log10)
    tf = np.zeros((D, V))
    for ind in range(D):
        for i in range(V):
            if termFreq[ind][i] > 0:
                tf[ind][i] = 1 + math.log(termFreq[ind][i], 10)
    del termFreq

    # find idf
    if 'Train' == dataset:
        # get df
        df = np.zeros((V, 1))
        for ind in range(D):
            for i in range(V):
                df[i] += termBin[ind][i]
        # get idf (log10(D/df))
        idf = np.zeros((V, 1))
        for i in range(V):
            if df[i] > 0:
                idf[i] = math.log(D, 10) - math.log(df[i], 10)
        del df
        np.save(tmpPath + '/idf_' + lStem + '.npy', idf)
    else:
        # if 'Test' == dataset, get idf from arguments.
        idf = np.load(tmpPath + '/idf_' + lStem + '.npy')
    del termBin

    # get tfidf
    tfidf = np.zeros((D, V))
    for ind in range(D):
        for i in range(V):
            tfidf[ind][i] = tf[ind][i] * idf[i]
    return tfidf

# class defination: feed forward neural network.
class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, dims):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.L1 = nn.Linear(dims, 20)
        self.L2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a1 = self.sigmoid(self.L1(x))
        a2 = self.sigmoid(self.L2(a1))
        return a2

# train the feed forward neural network.
def TrainFFNN(featTrain):
    '''
    train a feed forward neural network using train features.
    :param featTrain: train features - D * V
    :return: model - a FeedForwardNeuralNetwork object
    '''
    # initialize network weights with uniform distribution.
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight)
            nn.init.uniform_(m.bias)
    # calculate the train accuracy.
    def trainAccuracy(y, yhat):
        total = len(y)
        cnt = 0
        for i in range(total):
            err = y[i] - yhat[i]
            if abs(err) < 0.5:
                cnt += 1
        return cnt / total

    # get V and D.
    V = len(featTrain[0])
    D = len(featTrain)
    # sparse the corresponding label.
    dset = np.load(tmpPath + '/Train.npz', allow_pickle = True)
    labelTrain = dset['labelTrain']

    # shuffle the data and label.
    index = [i for i in range(D)]
    random.shuffle(index)
    featTrain = featTrain[index]
    labelTrain = labelTrain[index]

    # split the train and valid set.
    xTrain, xValid, yTrain, yValid = train_test_split(featTrain, labelTrain, test_size=0.2)
    # convert data (x,y) into tensor.
    xTrain = torch.Tensor(xTrain).cuda()
    yTrain = torch.LongTensor(yTrain).cuda()
    yTrain = yTrain.reshape(len(yTrain), 1)
    xValid = torch.Tensor(xValid).cuda()
    yValid = torch.LongTensor(yValid).cuda()
    yValid = yValid.reshape(len(yValid), 1)

    # convert to mini-batch form.
    batchsize = 256
    train = torchdata.TensorDataset(xTrain, yTrain)
    numTrain = len(train)
    trainloader = torchdata.DataLoader(train, batch_size=batchsize, shuffle=False)
    valid = torchdata.TensorDataset(xValid, yValid)
    numValid = len(valid)
    validloader = torchdata.DataLoader(valid, batch_size=batchsize, shuffle=False)

    # build the model of feed forward neural network.
    model = FeedForwardNeuralNetwork(V)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.apply(weight_init)
    model.to(device)
    # optimizing with stochastic gradient descent.
    optimizer = optim.SGD(model.parameters(), lr = 0.5)
    # seting loss function as mean squared error.
    criterion = nn.MSELoss()

    # run on each epoch.
    accList = [0]
    for epoch in range(10000):
        # training phase.
        model.train()
        lossTrain = 0
        accTrain = 0
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # set the gradients to zero.
            yhat = model.forward(data)  # get output
            loss = criterion(label.float(), yhat)
            loss.backward()
            optimizer.step()
            # statistic
            lossTrain += loss.item()
            preds = (yhat > 0.5).long()
            accTrain += torch.sum(torch.eq(preds, label).long()).item()
        lossTrain /= (iter + 1)
        accTrain *= 100 / numTrain

        # validation phase.
        model.eval()
        accValid = 0
        with torch.no_grad():
            for iter, (data, label) in enumerate(validloader):
                data = data.to(device)
                label = label.to(device)
                yhat = model.forward(data)  # get output
                # statistic
                preds = (yhat > 0.5).long()
                accValid += torch.sum(torch.eq(preds, label).long()).item()
        accValid *= 100 / numValid
        accList.append(accValid)

        # output information.
        if 0 == (epoch + 1) % 100:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%' % (
            epoch + 1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tmpPath + '/model.pth')
        # stop judgement.
        if (epoch + 1) >= 100 and accList[-1] < min(accList[-100:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tmpPath + '/model.pth'))
    return model

# test the feed forward neural network.
def TestFFNN(model, featTest):
    '''
    run test data using the feed forward neural network
    :param model: a FeedForwordNeuralNetwork object.
    :param featTest:  test features - D' * V
    :return: accuracy - 0~1
    :return: confusion - confusion matrix 2 * 2
    '''
    # get predictions for testing samples with model parameters.
    def GetPredictions(model, featTest):
        D = len(featTest)
        x = torch.Tensor(featTest).cuda()
        yhat = model.forward(x)
        predictions = np.zeros(D)
        for ind in range(D):
            if yhat[ind] > 0.5:
                predictions[ind] = 1
        return predictions

    # evaluate the predictions with gold labels, and get accuracy and confusion matrix.
    def Evaluation(predictions):
        # sparse the corresponding label.
        dset = np.load(tmpPath + '/Test.npz', allow_pickle = True)
        labelTest = dset['labelTest']
        D = len(labelTest)
        cls = 2
        # get confusion matrix.
        confusion = np.zeros((cls, cls))
        for ind in range(D):
            nRow = int(predictions[ind])
            nCol = int(labelTest[ind])
            confusion[nRow][nCol] += 1
        # get accuracy.
        accuracy = 0
        for ind in range(cls):
            accuracy += confusion[ind][ind]
        accuracy /= D
        return accuracy, confusion

    # get predictions for testing samples.
    predictions = GetPredictions(model, featTest)
    # get accuracy and confusion matrix.
    accuracy, confusion = Evaluation(predictions)
    return accuracy, confusion

# output the results.
def OutputFFNN(accuracy, confusion, lStem):
    '''
    output the results.
    :param accuracy: test accuracy 0~1
    :param confusion: confusion matrix 2 * 2
    :param lStem: stem setting - 'noStem', 'Stem'
    :return: none
    '''
    # input validation.
    if lStem not in ['noStem', 'Stem']:
        print('Error: stem setting invalid!')
        return
    # output on screen and to file.
    print('-------------------------------------------')
    print('Feed Forward Neural Network | ' + lStem )
    print('test accuracy : %.2f%%' % (accuracy * 100))
    print('confusion matrix :      (actual)')
    print('                    Neg         Pos')
    print('(predicted) Neg     %-4d(TN)    %-4d(FN)' % (confusion[0][0], confusion[0][1]))
    print('            Pos     %-4d(FP)    %-4d(TP)' % (confusion[1][0], confusion[1][1]))
    print('-------------------------------------------')
    return

# The program entrance.
if __name__ == "__main__":
    main()
