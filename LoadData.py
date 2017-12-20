from glob import glob
import shutil
import os,sys

dir_path = os.getcwd()
sys.path.append(dir_path)
data_path = os.path.join(dir_path,'database','output')
all_data_files = glob(os.path.join(data_path,'**','*.txt'),recursive=True)

