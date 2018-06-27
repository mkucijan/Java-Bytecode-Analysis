
from glob import glob
import json
import os
import random

PROJECT_DIR = os.path.split(os.getcwd())[0]
DB_PATH = os.path.join(PROJECT_DIR, 'TestParseOutput')
DIR_PATH = os.path.join(PROJECT_DIR, 's2s_byte2source', 'db')

TEST_FLAG = True
if TEST_FLAG:
    DB_PATH = os.path.join(PROJECT_DIR, 'test', 'tree')

TEST_SOURCES = os.path.join(DIR_PATH, 'test_sources.txt')
TEST_TARGETS = os.path.join(DIR_PATH, 'test_targets.txt')

VOCAB_SOURCE = os.path.join(DIR_PATH, 'vocab_source.txt')
VOCAB_TARGET = os.path.join(DIR_PATH, 'vocab_target.txt')

TRAIN_SOURCES = os.path.join(DIR_PATH, 'train_sources.txt')
TRAIN_TARGETS = os.path.join(DIR_PATH, 'train_targets.txt')

DEV_SOURCES = os.path.join(DIR_PATH, 'dev_sources.txt')
DEV_TARGETS = os.path.join(DIR_PATH, 'dev_targets.txt')

DIRS = [DIR_PATH, VOCAB_SOURCE, VOCAB_TARGET, TRAIN_SOURCES,
        TRAIN_TARGETS, DEV_SOURCES, DEV_TARGETS]

DEV_PARTITION = 0.1

try:
    os.makedirs(DIR_PATH)
except OSError as e:
    print("Folder",DIR_PATH,"already exists.")
    #raise e
    pass

class_path_names = glob(os.path.join(DB_PATH,'**','*.json'),recursive=True)

classes = {}
for class_path in class_path_names:
    class_name = os.path.split(class_path)[1].split('.json')[0]
    with open(class_path, 'r') as f:
        methods = json.load(f)
        if methods:
            classes[class_name] = methods
    
vocabulary_source = set()
vocabulary_target = set()

def serialize_tree(tree, instructions, statements):
    
    if 'instructions' in tree:
        statements.append('BEGIN_'+tree['name'])

    for child in tree['children']:
        serialize_tree(child, instructions, statements)

    if 'instructions' in tree:
        instructions += tree['instructions']
        statements.append('END_'+tree['name'])

hash_values = []

if TEST_FLAG:
    with open(TEST_SOURCES, 'w') as src, open(TEST_TARGETS, 'w') as trg:
        for methods in classes.values():
            for method in methods.values():
                
                instructions, statements = [], []
                serialize_tree(method, instructions, statements)
                
                if instructions and statements:
                    src.write(' '.join(instruction[1] for instruction in instructions)+'\n')
                    trg.write(' '.join(statement for statement in statements)+'\n')
else:
    with open(TRAIN_SOURCES, 'w') as src, open(TRAIN_TARGETS, 'w') as trg, open(DEV_SOURCES, 'w') as dev_src, open(DEV_TARGETS, 'w') as dev_trg:
        for methods in classes.values():
            for method in methods.values():
                
                instructions, statements = [], []
                serialize_tree(method, instructions, statements)
                instructions_2 = []
                for instruction in instructions:
                    instruction[1] = instruction[1].replace('\r','').replace('\n','')
                    if not ' ' in instruction[1] or not '' is instruction[1]:
                        if not ('java' in instruction[1] or 'org' in instruction[1]):
                            vocabulary_source.add(instruction[1])
                            hash_values.append(hash(instruction[1]))
                    if instruction[1] in vocabulary_source:
                        instructions_2.append(instruction)
                instructions = instructions_2
                for statement in statements:
                    vocabulary_target.add(statement)
                
                if instructions and statements:
                    if  random.random() < DEV_PARTITION:
                        dev_src.write(' '.join(instruction[1] for instruction in instructions)+'\n')
                        dev_trg.write(' '.join(statement for statement in statements)+'\n')
                    else:
                        src.write(' '.join(instruction[1] for instruction in instructions)+'\n')
                        trg.write(' '.join(statement for statement in statements)+'\n')

    with open(VOCAB_SOURCE, 'w') as src, open(VOCAB_TARGET, 'w') as trg:
        for instr in vocabulary_source:
            src.write(instr+'\n')
        for statement in vocabulary_target:
            trg.write(statement+'\n')
            
        
