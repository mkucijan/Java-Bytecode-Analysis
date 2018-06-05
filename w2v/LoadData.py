from glob import glob
import json
import os,sys
import pickle


dir_path = os.getcwd()
sys.path.append(dir_path)
data_path = os.path.join(dir_path,'output')
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

pickle_out=open('cache/database.dict','wb')
pickle.dump(classes, pickle_out)
pickle_out.close()