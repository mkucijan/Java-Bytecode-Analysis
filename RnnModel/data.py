"""
Classes for processing instructions and its operands into a representation useable for the model
"""

from glob import glob
from random import shuffle
import json
import os
import pickle
import re
import sys
import math

import numpy as np

from RnnModel import logger


DIR_PATH = os.path.split(os.getcwd())[0]
sys.path.append(DIR_PATH)
#logger.info(DIR_PATH)

from JavaClassParser import ByteCode

class Data(object):

    """
    Default class for handling database of instructions labeled with statements from
    source code.
    
    Classes:
    ['default', 'if', 'else', 'while', 'for', 'switch']

    """

    def __init__(self, X = None, Y = None):
        self.X = X
        self.Y = Y
        self.indices = None
        if not X is None:
            self.indices = list([i for i in range(self.length)])
        self.partition_indices = self.indices
        self.startIndex = 0
    
    def __str__(self):
        return self.Y
    
    def loadDataFromJSON(self, data_path = os.path.join(DIR_PATH, 'output'),
                         save=False, loadMatrix=True):

        """
        Data is saved in JSON files holding one class with all of its methods per
        JSON file, methods hold instructions and operands with its respective labels.
        """
        
        all_data_files = glob(os.path.join(data_path,'**','*.json'),recursive=True)

        classes={}
        for f in all_data_files:
            name = os.path.split(f)[-1][:-5]
            f = open(f,'r')
            methods = {}
            jsonFile = json.load(f)
            for method in jsonFile:
                methods[method] = {'x':[], 'y':[], 'index':[]}
                for byteIndex in jsonFile[method]:
                    methods[method]['x'].append(jsonFile[method][byteIndex][0])
                    methods[method]['y'].append( (jsonFile[method][byteIndex][1],
                                                  jsonFile[method][byteIndex][2]) )
                    methods[method]['index'].append(int(byteIndex))

            if methods:
                classes[name] = methods

        if save:
            pickle_out=open(os.path.join(DIR_PATH, 'cache', 'database.dict'), 'wb')
            pickle.dump(classes, pickle_out)
            pickle_out.close()

        self.classes = classes
        X_train = []
        Y_train = []
        self.dictionary = {'padding':0, 'UKN':1}
        instr_index = 2
        if loadMatrix:
            for dclass in classes.values():
                for method in dclass.values():
                    instructions = method['x']
                    for instruction in instructions:
                        if instruction not in self.dictionary:
                            self.dictionary[instruction] = instr_index
                            instr_index += 1      
                    labels = method['y']
                    byteIndex = method['index']
                    
                    X_train.append(instructions)
                    Y_train.append(labels)
    
        self.__init__(X_train, Y_train)

    def filterOperands(self, overwrite = True):
        """
        Removes operand and its labels from data leaving only instructions
        """

        new_X = []
        new_Y = []
        for x,y in zip(self.X, self.Y):
            seq_X = []
            seq_Y = []
            for seq_x, seq_y in zip(x,y):
                if seq_x in ByteCode.mnemonicMap:
                    seq_X.append(seq_x)
                    seq_Y.append(seq_y)

            new_X.append(seq_X)
            new_Y.append(seq_Y)

        if overwrite:
            self.X = new_X
            self.Y = new_Y

        return new_X, new_Y

    def relabelData(self, matchLabel = 1, matched = 1, unmatched = 0,   
                    overwrite = False):

        new_Y = []
        for y_seq in self.Y:
            y_seq = np.array(y_seq)
            y_seq = list(map(
                    lambda y: (y[0], matched) if (y[1]==matchLabel) else (y[0], unmatched),
                    y_seq))
            
            new_Y.append(y_seq)

        if overwrite:
            self.Y = new_Y

        return self.X, new_Y

    @property
    def baseline(self):
        sum_zero = num = 0
        for y_seq in self.Y:
            for y in y_seq:
                sum_zero += 1 if y[1]==0 else 0
                num += 1
        
        return sum_zero/num
    
    def simplifySignature(self, overwrite = False):
        """
        Filters method signatures to be represented only by its name.
        """    

        regex = re.compile(
            r'^(.+)/([^/]+)\.([\w\$]+\$)?(?P<method_name>[^:\$]+):([(](?P<args>[^)]*)[)])?(?P<return>.+)'
            )

        new_X = []
        for x_seq in self.X:
            new_seq = []
            for x in x_seq:
                m = regex.match(x)
                if m:
                    new_seq.append(m.group('method_name'))
                else:
                    new_seq.append(x)
            new_X.append(new_seq)

        if overwrite:
            self.X = new_X

        return new_X, self.Y

    def filterNoJumps(self, overwrite = True):
        """
        Filter methods without jumps which are mandatory for containing if or while statements
        """
        new_X = []
        new_Y = []
        keys = ['if', 'goto']
        for x_seq, y_seq in zip(self.X, self.Y):
            if any([key in instr for key in keys for instr in x_seq]):
                new_X.append(x_seq)
                new_Y.append(y_seq)

        if overwrite:
            self.X = new_X
            self.Y = new_Y

        return new_X, new_Y
            
    @classmethod
    def deserialize(cls, directory):
        new = cls()
        new.loadDataFromJSON(directory)
        return new

    @property
    def output_size(self):
        return max([max(np.array(y)[:,1]) for y in self.Y])+1

    @property
    def length(self):
        return len(self.X)

    @property
    def vocabulary_size(self):
        return len(self.vocabulary) 
    
    def shuffle(self):
        indices = shuffle(list(range(self.length)))
        self.indices = indices
    
    def __getitem__(self, partition):
        try:
            assert(partition < 1.0)
        except AssertionError as e:
            if partition == 1.0:
                self.startIndex = 0
                self.partition_indices = self.indices
                return self
            else:
                raise e
        index = int(partition*self.length)
        try:
            assert(index+self.startIndex <= self.length)
        except AssertionError as e:
            self.startIndex = 0
            logger.info("Parition overflow, spliting data from start.") 
        self.partition_indices = [
            self.indices[i] for i in range(self.startIndex, self.startIndex+index)
            ]
        self.startIndex += index

        return self
    
    def getPartition(self, partition):
        self[partition]
        X=[]
        Y=[]
        for i in self.partition_indices:
            X.append(self.X[i])
            Y.append(self.Y[i])
        
        return DataPartition(X, Y, self.output_size)
    

    #WRONG TIME_STEPS NOT N_STEPS
    def epoch(self, time_steps, batch_size):
        X = np.array(self.X)
        Y = np.array(self.Y)
        indices = np.array(self.partition_indices)
        startIndex = 0
        for i in range(time_steps):
            if (startIndex+batch_size)>len(indices):
                startIndex = 0
            batch_indices = indices[startIndex:startIndex+batch_size]
            startIndex += batch_size
            yield True, X[batch_indices], Y[batch_indices], startIndex/len(indices)
    
    @staticmethod
    def flattenData(X, Y, time_steps):

        new_X = []
        new_Y = []
        for x,y in zip(X,Y):
            if len(x)<=time_steps:
                new_X.append(x)
                new_Y.append(y)
            else:
                for i in range(int(len(x)/time_steps)):
                    new_X.append(x[i*time_steps:(i+1)*time_steps])
                    new_Y.append(y[i*time_steps:(i+1)*time_steps])
                if x[(i+1)*time_steps:]:
                    new_X.append(x[(i+1)*time_steps:])
                    new_Y.append(y[(i+1)*time_steps:])
        
        return new_X, new_Y

class DataPartition(object):

    def __init__(self, X, Y, output_size):
        self.X = X
        self.Y = Y
        self.output_size = output_size
        self.getVocabulary()

    def getVocabulary(self):
        vocabulary = {'padding':0, 'UKN': 1}
        instr_index = 2
        for X_seq in self.X:
            for instruction in X_seq:
                if not instruction in vocabulary:
                    vocabulary[instruction] = instr_index
                    instr_index += 1
        
        self.vocabulary = vocabulary
        return vocabulary
    
    def epoch(self, time_steps, batch_size, flatten=True):

        if flatten:
            self.X, self.Y = Data.flattenData(self.X, self.Y, time_steps)
        
        X = np.array(self.X)
        Y = np.array(self.Y)
        '''
        lens = [len(x_seq),i for i,x_seq in enumerate(X)]
        indices = np.array(sorted(lens, lambda len_seq, index_seq : len_seq, reverse=True))[:,1]
        
        X = X[indices]
        Y = Y[indices]

        #CUTOUT TOO LONG METHODS by batch_size
        for i, x in enumerate(X):
            if len(x)/time_steps > batch_size:
                x = x[:batch_size*time_steps]
                X[i] = x
                Y[i] = Y[i][:batch_size*time_steps]
                # TODO: overflowed put back in dataset?
        
        '''
        data_index = 0
        n_steps = math.ceil(len(X)/batch_size)

        for k in range(n_steps):
            batch_X = np.zeros((batch_size, time_steps))
            batch_Y = np.zeros((batch_size, time_steps, self.output_size))
            seq_len = np.zeros((batch_size))
            mask = np.zeros((batch_size, time_steps, self.output_size))
            for i in range(batch_size):
                if data_index<len(X):
                    for j in range(len(X[data_index])):
                        batch_X[i,j] = self.vocabulary.get(X[data_index][j], 1)
                        batch_Y[i,j,int( Y[data_index][j][1] )] = 1
                        mask[i,j] = 1
                    seq_len[i] = len(X[data_index])
                    data_index += 1
            yield True, batch_X, batch_Y, seq_len, mask, k/n_steps


    @property
    def vocabulary_size(self):
        return len(self.vocabulary) 