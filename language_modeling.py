'''
  Author: Julia Jeng, Shu Wang, Arman Anwar
  Brief: AIT 726 Homework 2
  Usage:
      Put file 'language_modeling.py' and folder 'twitter' in the same folder.
  Command to run:
      python language_modeling.py
  Description:
      Build and train a feed forward neural network (FFNN) with 2 layers with hidden vector size 20.
      Initalized weights: random weights.
      Loss function: mean squared error.
      Activation function: sigmoid.
      Learning rate: 0.0001-0.00001.
      Train/valid rate: 4:1
      Emoticon tokenizer: TweetTokenizer
'''

import os
import re
import sys
import random
from random import choice
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from nltk import word_tokenize
from itertools import chain
from collections import defaultdict
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

# global path
logPath = './language_modeling.txt'
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
    if (os.path.exists(tmpPath + '/dataTrain.npy') and
        os.path.exists(tmpPath + '/dataTest.npy') and
        os.path.exists(tmpPath + '/vocab.npy')):
        dataTrain = np.load(tmpPath + '/dataTrain.npy', allow_pickle = True)
        dataTest = np.load(tmpPath + '/dataTest.npy', allow_pickle = True)
        vocab = np.load(tmpPath + '/vocab.npy', allow_pickle = True)
        print('Successfully load data from \'' + tmpPath + '\'.')
    else:
        dataTrain, dataTest, vocab = CreateVocabulary()

    # Extract training ngrams.
    if (os.path.exists(tmpPath + '/gramTrain.npy') and
            os.path.exists(tmpPath + '/labelTrain.npy')):
        gramTrain = np.load(tmpPath + '/gramTrain.npy', allow_pickle = True)
        labelTrain = np.load(tmpPath + '/labelTrain.npy', allow_pickle = True)
        print('Successfully load n-grams from \'' + tmpPath + '\'.')
        print('N-gram train data: positive %d, negative %d.' % (sum(labelTrain), len(labelTrain) - sum(labelTrain)))
    else:
        gramTrain, labelTrain = ExtractNGram(dataTrain, vocab)
        np.save(tmpPath + '/gramTrain.npy', gramTrain)
        np.save(tmpPath + '/labelTrain.npy', labelTrain)

    # train the FFNN model.
    model = TrainFFNN(gramTrain, labelTrain, vocab)

    # Extract testing ngrams.
    if (os.path.exists(tmpPath + '/gramTest.npy') and
            os.path.exists(tmpPath + '/labelTest.npy')):
        gramTest = np.load(tmpPath + '/gramTest.npy', allow_pickle = True)
        labelTest = np.load(tmpPath + '/labelTest.npy', allow_pickle = True)
        print('Successfully load n-grams from \'' + tmpPath + '\'.')
        print('N-gram test data: positive %d, negative %d.' % (sum(labelTest), len(labelTest) - sum(labelTest)))
    else:
        gramTest, labelTest = ExtractNGram(dataTest, vocab)
        np.save(tmpPath + '/gramTest.npy', gramTest)
        np.save(tmpPath + '/labelTest.npy', labelTest)

    TestFFNN(model, gramTest, labelTest)

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

    # if there is no 'tmp' folder, create one.
    if not os.path.exists(tmpPath):
        os.mkdir(tmpPath)
    if not os.path.exists(datPath):
        print('[ERROR]: Cannot find the data path \'' + datPath + '\'.')

    # read the training data.
    dataTrain = []
    for root, ds, fs in os.walk(datPath + '/train/positive/'):
        for file in fs:
            # get the file path.
            fullname = os.path.join(root, file)
            # get the training data.
            data = open(fullname, encoding = "utf8").read()
            # preprocess the data.
            data = Preprocess(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTrain.append(tokens)
            #print(tokens)
    print('Load TrainSet: %d samples.' % len(dataTrain))
    np.save(tmpPath + '/dataTrain.npy', dataTrain)

    # build the vocabulary from training set.
    vocab = list(set(list(chain.from_iterable(dataTrain))))
    print('Vocabulary: %d tokens.' % len(vocab))
    np.save(tmpPath + '/vocab.npy', vocab)

    dataTest = []
    for root, ds, fs in os.walk(datPath + '/test/positive/'):
        for file in fs:
            # get the file path.
            fullname = os.path.join(root, file)
            # get the training data.
            data = open(fullname, encoding = "utf8").read()
            # preprocess the data.
            data = Preprocess(data)
            # get the tokens for the data.
            tokens = GetTokens(data)
            dataTest.append(tokens)
            #print(tokens)
    print('Load TestSet: %d samples.' % len(dataTest))
    np.save(tmpPath + '/dataTest.npy', dataTest)
    return dataTrain, dataTest, vocab

# extract gram and label for data using vocab
def ExtractNGram(data, vocab):
    '''
    Extract 2-grams for dataset
    :param data: data set D * (tokens)
    :param vocab: vocabulary 1 * V
    :return: gramData gramLabel in numpy.array
    '''
    # generate ngram list and label.
    ngramList = []
    ngramLabel = []
    # get all positive 2-grams.
    posList = []
    for doc in data:
        for gram in ngrams(doc, 2):
            # gram is a 2-tuple ('A', 'B'), ('A', 'C').
            posList.append(gram)
            ngramList.append([gram[0], gram[1]])
            ngramLabel.append([1])

    # create negative 2-grams.
    ngramDict = defaultdict(list)
    for gram in posList:
        # ngramdict is a list dictionary {'A': ['B', 'C']}.
        ngramDict[gram[0]].append(gram[1])
    for gram in posList:
        for i in range(2):
            # randomly sample the second word.
            while True:
                word = choice(vocab)
                if (word != gram[0]) and (word not in ngramDict[gram[0]]):
                    break
            ngramList.append([gram[0], word])
            ngramLabel.append([0])
    numPos = len(posList)
    num = len(ngramList)
    print('N-gram data: positive %d, negative %d.' % (numPos, num-numPos))

    # build vocabulary dictionary.
    vocabDict = {word: i for i, word in enumerate(vocab)}
    # convert ngram to index
    gramData = []
    gramLabel = []
    for ind in range(0, num):
        ngram = ngramList[ind]
        if (ngram[0] in vocab and ngram[1] in vocab):
            gramData.append([vocabDict[word] for word in ngramList[ind]])
            gramLabel.append(ngramLabel[ind])
    return np.array(gramData), np.array(gramLabel)

# class of LanguageModeling
class LanguageModeling(nn.Module):
    def __init__(self, V):
        super(LanguageModeling, self).__init__()
        self.dims = 20  # embedding dimension 20-200
        self.embedding = nn.Embedding(V, self.dims)
        self.L1 = nn.Linear(2 * self.dims, 20)
        self.L2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        proj = self.embedding(x).view((-1, 2 * self.dims)) # row vector.
        a1 = self.sigmoid(self.L1(proj))
        a2 = self.sigmoid(self.L2(a1))
        return a2

# train the feed forward neural network.
def TrainFFNN(gramTrain, labelTrain, vocab):
    '''
    train the FFNN with gramTrain, labelTrain and vocab
    :param gramTrain: data set N * 2
    :param labelTrain: label N * 1
    :param vocab: 1 * V
    :return:
    '''
    # initialize network weights with uniform distribution.
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight)
            nn.init.uniform_(m.bias)

    # vocabulary number.
    V = len(vocab)
    # sample number.
    N = len(labelTrain)

    # shuffle the data and label.
    index = [i for i in range(N)]
    random.shuffle(index)
    gramTrain = gramTrain[index]
    labelTrain = labelTrain[index]

    # split the train and valid set.
    xTrain, xValid, yTrain, yValid = train_test_split(gramTrain, labelTrain, test_size = 0.2)
    # convert data (x,y) into tensor.
    xTrain = torch.LongTensor(xTrain).cuda()
    yTrain = torch.LongTensor(yTrain).cuda()
    xValid = torch.LongTensor(xValid).cuda()
    yValid = torch.LongTensor(yValid).cuda()

    # convert to mini-batch form
    batchsize = 200
    train = torchdata.TensorDataset(xTrain, yTrain)
    numTrain = len(train)
    trainloader = torchdata.DataLoader(train, batch_size = batchsize, shuffle = False)
    valid = torchdata.TensorDataset(xValid, yValid)
    numValid = len(valid)
    validloader = torchdata.DataLoader(valid, batch_size = batchsize, shuffle = False)

    # build the model of feed forward neural network.
    model = LanguageModeling(V)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.apply(weight_init)
    model.to(device)
    # optimizing with stochastic gradient descent.
    optimizer = optim.SGD(model.parameters(), lr = 0.25)
    # seting loss function as mean squared error.
    criterion = nn.MSELoss()

    # run on each epoch.
    accList = [0]
    for epoch in range(1000):
        # training phase.
        model.train()
        lossTrain = 0
        accTrain = 0
        for iter, (data, label) in enumerate(trainloader):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad() # set the gradients to zero.
            yhat = model.forward(data) # get output
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
        if 0 == (epoch + 1) % 10:
            print('[Epoch %03d] loss: %.3f, train acc: %.3f%%, valid acc: %.3f%%' % (epoch+1, lossTrain, accTrain, accValid))
        # save the best model.
        if accList[-1] > max(accList[0:-1]):
            torch.save(model.state_dict(), tmpPath + '/model.pth')
        # stop judgement.
        if (epoch+1) >= 10 and accList[-1] < min(accList[-10:-1]):
            break

    # load best model.
    model.load_state_dict(torch.load(tmpPath + '/model.pth'))
    return model

# test the feed forward neural network.
def TestFFNN(model, gramTest, labelTest):
    '''
    test the gramTest with model, and output the accuracy
    :param model: FFNN model
    :param gramTest: data set N * 2
    :param labelTest: label N * 1
    :return: accuracy - testing accuracy
    '''
    # prepare for model.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # prepare for data.
    data = torch.LongTensor(gramTest).cuda()
    label = torch.LongTensor(labelTest).cuda()
    numTest = len(label)

    # test phase.
    data = data.to(device)
    label = label.to(device)
    yhat = model.forward(data)  # get output

    # statistic
    preds = (yhat > 0.5).long()
    numCorrect = torch.sum(torch.eq(preds, label).long()).item()
    accTest = 100 * numCorrect / numTest
    print('-------------------------------------------')
    print('Test accuracy: %.3f%%' % (accTest))
    print('-------------------------------------------')
    return accTest

# The program entrance.
if __name__ == "__main__":
    main()