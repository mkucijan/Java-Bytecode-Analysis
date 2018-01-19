from glob import glob
import shutil
import os,sys
import pickle


dir_path = os.getcwd()
sys.path.append(dir_path)
data_path = os.path.join(dir_path,'database','output')
all_data_files = glob(os.path.join(data_path,'**','*.txt'),recursive=True)

classes={}
for f in all_data_files:
    name = os.path.split(f)[-1][:-4]
    f = open(f,'r')
    methods = {}
    for line in f.readlines():
        if 'Method' in line:
            method = line
            methods[method] = {'x':[], 'y':[], 'index':[]}
            continue
        line = line.split()
        if len(line)==4:
            byteIndex = int(line[0])
            instruction = line[1]
            embendingLayer = int(line[2][1:-1])
            embendingType = int(line[3][:-1])
            methods[method]['x'].append(instruction)
            methods[method]['y'].append((embendingLayer,embendingType))
            methods[method]['index'].append(byteIndex)
    
    if methods:
        classes[name] = methods

pickle_out=open('cache/database.dict','wb')
pickle.dump(classes, pickle_out)
pickle_out.close()