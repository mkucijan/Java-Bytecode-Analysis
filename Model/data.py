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

import binascii
import numpy as np

from Model import logger


DIR_PATH = os.path.split(os.getcwd())[0]
sys.path.append(DIR_PATH)
#logger.info(DIR_PATH)

from JavaClassParser import ByteCode


class Encode(object):
    
    def __init__(self, num_bits = 16, encode_type='bin'):
        self.num_bits = num_bits
        self.zero_value = 2**(num_bits-1)
        
        if encode_type is 'bin':
            self.encode = self.bin_code
        elif encode_type is 'gray':
            self.encode = self.gray_code

    def gray_code(self, n):
        n -= 1
        return format(n^(n >> 1), '0'+str(self.num_bits)+'b')

    def bin_code(self, n):
        return format(n, '0'+str(self.num_bits)+'b')

    def encode_arg(self, value):
        n=self.zero_value+value
        return np.array(list(map(lambda x: float(x), list(self.encode(n)))))
    
    def from_hex(self, value_hex):
        return self.encode_arg(int(value_hex, 16))      

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
        if X:
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

    def relabelData(self, matchLabel = [1], matched = 1, unmatched = None,   
                    overwrite = False):

        def relabel(y):
            if y[1] in matchLabel:
                return (y[0], matched)
            else:
                if unmatched:
                    return (y[0], unmatched)
                else:
                    return y

        new_Y = []
        for y_seq in self.Y:
            y_seq = np.array(y_seq)
            y_seq = list(map(relabel, y_seq))
            
            new_Y.append(y_seq)

        if overwrite:
            self.Y = new_Y

        return self.X, new_Y
    
    def relabel_by_embedding(self, overwrite=False):
        
        new_Y = []
        for y_seq in self.Y:
            embed_dict = {}
            new_y_seq = []
            for y in y_seq:
                if y[0] in embed_dict:
                    new_type = embed_dict[y[0]]
                else:
                    embed_dict[y[0]] = y[1]
                    new_type = y[1]
                new_y_seq.append((y[0],new_type))
            new_Y.append(new_y_seq)

        if overwrite:
            self.Y = new_Y
        
        return new_Y
                    

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
                try:
                    m = regex.match(x)
                except Exception as e:
                    m = None
                if m:
                    new_seq.append(m.group('method_name'))
                else:
                    new_seq.append(x)
            new_X.append(new_seq)

        if overwrite:
            self.X = new_X

        return new_X, self.Y

    def filter_no_jumps(self, overwrite = True):
        """
        Filter methods without jumps which are mandatory for containing if or while statements
        """
        new_X = []
        new_Y = []
        keys = ['if', 'goto']
        for x_seq, y_seq in zip(self.X, self.Y):
            try:
                if any([key in str(instr) for key in keys for instr in x_seq]):
                    new_X.append(x_seq)
                    new_Y.append(y_seq)
            except Exception:
                raise ValueError

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
        self.indices = list(range(self.length))
        shuffle(self.indices)
    
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
        self.flattened = False
    
    @property
    def baseline(self):
        sum_zero = num = 0
        for y_seq in self.Y:
            for y in y_seq:
                sum_zero += 1 if y[1]==0 else 0
                num += 1
        
        return sum_zero/num

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
    
    def seperate_args(self):
        X = []
        Y = []
        X_args = []
        for x_seq, y_seq in zip(self.X, self.Y):
            new_x = []
            new_x_arg = []
            new_y = []
            last_instr = False
            i=0
            while(i<len(x_seq)):
                if last_instr:
                    if not x_seq[i] in ByteCode.mnemonicMap:    
                        new_x_arg.append(x_seq[i])
                        i += 1
                    else:
                        new_x_arg.append("")
                    last_instr = False
                    continue
                if x_seq[i] in ByteCode.mnemonicMap:
                    new_x.append(x_seq[i])
                    new_y.append(y_seq[i])
                    last_instr = True
                    i += 1
            if last_instr:
                new_x_arg.append('')
            assert len(new_x) == len(new_x_arg)
            
            X.append(new_x)
            X_args.append(new_x_arg)
            Y.append(new_y)

        self.X = X
        self.X_args = X_args
        self.Y = Y

        return X, X_args, Y
    
    def push_long_seq(self, time_steps):
        
        X = []
        Y = []
        X_long = []
        Y_long = []
        for x,y in zip(self.X, self.Y):
            if len(x)>time_steps:
                X_long.append(x)
                Y_long.append(y)
            else:
                X.append(x)
                Y.append(y)
        
        self.X = X + X_long
        self.Y = Y + Y_long

    
    def epoch(self, time_steps, batch_size, vocabulary=None, flatten=True, args_dim = None, encode_int=True):

        if flatten and not self.flattened:
            self.X, self.Y = Data.flattenData(self.X, self.Y, time_steps)
            self.flattened = True
            # TODO change this, and fix state feeding

        if vocabulary is None:
            vocabulary = self.vocabulary

        if args_dim:
            self.seperate_args()
            encode_fun = Encode(args_dim, 'bin').from_hex
        
        if encode_int:
            encode_fun_int = Encode(32, 'bin').encode_arg
        
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

            instruction_values = []
            if encode_int:
                batch_encoded_int = np.zeros((batch_size, time_steps, 32))
                num_ints = 0

            if args_dim:
                batch_X_args = np.zeros((batch_size, time_steps, args_dim))

            for i in range(batch_size):
                if data_index<len(X):
                    num_iter = min(len(X[data_index]), time_steps)
                    instr_index = 0
                    instruction_values_seq = []
                    for j in range(num_iter):
                        if encode_int:
                            try:
                                value = int(X[data_index][j])
                                num_ints += 1
                                batch_encoded_int[i, instr_index] = encode_fun_int(value)
                                continue
                            except Exception as e:
                                pass
                        batch_X[i,instr_index] = vocabulary.get(X[data_index][j], 1)
                        instruction_values_seq.append(X[data_index][j])
                        batch_Y[i, instr_index,int( Y[data_index][j][1] )] = 1
                        mask[i,instr_index] = 1
                        seq_len[i] += 1
                        if args_dim:
                            batch_X_args[i, instr_index] = encode_fun(self.X_args[data_index][j])
                        instr_index += 1
                    instruction_values.append(instruction_values_seq)
                    data_index += 1
            
            if args_dim:
                yield True, (batch_X, batch_X_args), batch_Y, seq_len, mask, k/n_steps, instruction_values
            elif encode_int:
                yield True, (batch_X, batch_encoded_int), batch_Y, seq_len, mask, k/n_steps, instruction_values  
            else:
                yield True, batch_X, batch_Y, seq_len, mask, k/n_steps, instruction_values

    def encode(self, x_seq):
        ret = []
        for x in x_seq:
            ret.append(self.vocabulary.get(x,1))
        return ret


    @property
    def vocabulary_size(self):
        return len(self.vocabulary)


