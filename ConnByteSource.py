import argparse
import subprocess
import re
import sys
from glob import glob
import os,errno
import shutil

def getLineTable(jf_name):
    linesCommand = "javap -l "+jf_name
    process = subprocess.Popen(linesCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('utf-8')
    return output

def getMethodsLines(jf_name):
    disassembleCommand = "javap -c "+jf_name
    process = subprocess.Popen(disassembleCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('utf-8')
    return output

def createLineMap(jf_name):
    
    methods = {}
    output = getLineTable(jf_name)
    output = output.split('\n')
    min_line = 10000000
    max_line = 0
    code = {}
    inmethod=False
    method_name = ""
    for i in range(len(output)):
        if 'LineNumberTable:' in output[i]:
            method_name=output[i-1]
            min_line = 10000000
            max_line = 0
            code = {}
        elif re.match("\s*line \d+: \d+\s*",output[i]):
            source_line,method_line=tuple(re.findall(r'\d+',output[i]))
            source_line,method_line = int(source_line),int(method_line)
            code[method_line]=source_line
            min_line=min(min_line, source_line)
            max_line=max(max_line, source_line)
            inmethod=True
        elif inmethod:
            inmethod=False
            methods[method_name]={
                'code':code,
                'max':max_line,
                'min':min_line
            }

        else:    
            continue
    
    return methods

def generateOutputLine(instruction,mnemonicMap):
    mnemonic = instruction[1]
    opCount = mnemonicMap[mnemonic].getOpCodeCount()

def mapBytecodeToLineNumber(jf_name, methods):
    mnemonicMap = ByteCode.mnemonicMap
    
    code_by_lines = {}
    output=getMethodsLines(jf_name)
    output=output.split('\n')
    method_name = ""
    cur_line = 0
    switch_true = False
    cmnd = ""
    for i in range(len(output)):
        if 'Code:' in output[i]:
            method_name = output[i-1]
        elif switch_true:
            cmnd += output[i]
            if '}' in output[i]:
                switch_true = False
                code_by_lines[cur_line] += cmnd + "\n"
            
        elif re.match('\s*\d+:.*',output[i],re.DOTALL):
            if 'lookupswitch' in output[i] or tableswitch:
                switch_true = True
                cmnd = output[i]
                continue
            instruction=output[i].split(':')
            method_line, cmnd = instruction[0],

            method_line = int(method_line)
            if method_line in methods[method_name]['code']:
                cur_line = methods[method_name]['code'][method_line]
                code_by_lines[cur_line] = ""
            code_by_lines[cur_line] += cmnd + "\n"
        else:
            continue

    return code_by_lines
def bindCode(jf_name):
    
    methods = createLineMap(jf_name)

    code_by_lines = mapBytecodeToLineNumber(jf_name, methods)

    return code_by_lines

def main():
    parser = argparse.ArgumentParser(description="Script for binding java source code "+
                                                "to its bytecode representation using javap.")
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-v", "--verbose", action="store_true")
    #group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-cp", "--classpath",nargs='?',default=sys.path[0], 
                        help="Classpath for compiling java code")
    parser.add_argument("-f", "--files", nargs='*',
                        help="Java classes for which to generate output, "+
                        "by default generates for all classes in classpath")
    parser.add_argument("-d", "--directory", nargs='?', default=sys.path[0]+'/.class_files', 
                        help="Directory for compiled class files, use -r to auto remove.")
    parser.add_argument("-r", "--remove", action="store_true")
    parser.add_argument("-o", "--output", nargs='?', default=sys.path[0]+'/output', 
                        help="Output directory")
    args = parser.parse_args()

    class_path = args.classpath
    java_files = args.files
    save_class_directory = args.directory
    output_folder = args.output

    try:
        os.makedirs(output_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    try:
        os.makedirs(save_class_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    '''
    class_path = "/home/yolkin/NetBeansProjects/Cvor/src"
    java_files = ["senzor/udp/UDPClient.java"]
    save_class_directory = sys.path[0]+"/.class_files"
    '''


    
    if not java_files:
        java_files = glob(class_path+'/**/*.java',recursive=True)


    compileCommand = "javac -cp " + class_path + " -d "+ save_class_directory + " -g " + \
                     ' '.join(java_files) 
    process = subprocess.Popen(compileCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode('utf-8')

    for java_file in glob(save_class_directory+"/**",recursive=True):
        if java_file.endswith(".class"):
            
            jf_name =java_file[:-6].split(save_class_directory)[1]
            code_by_lines = bindCode(java_file)

            filename = output_folder+jf_name+".txt"
            
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise


            f=open(filename,'w')
            for key in sorted(code_by_lines.keys()):
                f.write("Line: "+str(key)+"\n")
                f.write(code_by_lines[key]+"\n") 
                #print("Line:",key)
                #print(code_by_lines[key])
            f.close()

    #links = re.search('LineNumberTable(.*)LocalVariableTable',output,re.DOTALL).group(1)
    #print(links)

    if args.remove:
        print("Deleting .class files...")
        shutil.rmtree(save_class_directory)
        #map(os.remove, filter(lambda url: url.endswith('.class'),
        #    glob(save_class_directory+"/**",recursive=True)))
        

if __name__ == '__main__':
    main()