import argparse
import subprocess
import sys
from glob import glob
import os,errno
import shutil

from JavaClassParser.Parser import Parser

keywords = [
    'if',
    'else',
    'while',
    'for',
    'switch'
]

def bindCode(class_file, source):
    parser = Parser(class_file)
    parser.parse()
    codeByLines = dict()
    file = open(source)
    source = file.readlines()

    for method in parser.methods:
        try:
            getattr(method,'code')
        except AttributeError:
            if 'abstract' not in str(method):
                print(method)
            continue
        methodLines = dict()
        instructions = method.code.instructions
        lineNumberTable = method.code.lineNumberTable.code2Line
        
        if ('<init>' in method.name) and (len(lineNumberTable) is 1):
            continue    #default constrcutor

        numEmbending = 0
        lastIndex = lineNumberTable[instructions[0].startIndex] -1  # instructions[0].startIndex = 0 always
                                                                    # -1 because indexing starts with 1
        embendingType = [0]
        embendingLayer = [0]
        for instruction in instructions:
            byteIndex = instruction.startIndex
            if byteIndex in lineNumberTable:
                lineNumber = lineNumberTable[byteIndex]
                for i in range(lastIndex,lineNumber):

                    try:
                        currentLine = source[i]
                    except Exception as e:
                        return codeByLines
                    openBrackets = currentLine.count('{')
                    closedBrackets = currentLine.count('}')
                    diff = openBrackets - closedBrackets
                    numEmbending +=  diff
                    numEmbending = max(0,numEmbending)
                    added = False
                    for i in range(len(keywords)):
                        if keywords[i] in currentLine:
                            embendingType.append(i+1)
                            added = True
                            embendingLayer.append(numEmbending)
                            break
                    i=len(embendingLayer)-1
                    while(i>=0):
                        if embendingLayer[i]>numEmbending:
                            embendingLayer.pop()
                            embendingType.pop()
                            i -= 1
                        else:
                            break
                lastIndex = lineNumber
            bytecode = instruction.mnemonic
            methodLines[str(byteIndex) + " " + bytecode] = (len(embendingType)-1,embendingType[-1])
            offset = 1
            if 'wide' in instruction.mnemonic:
                methodLines[str(byteIndex+offset) + " " + instruction.opcode.getMnemonic()] = (len(embendingType)-1,embendingType[-1])
                offset += 1
            if instruction.args:
                for arg, fromPool, frm, size in zip(instruction.argValues,instruction.constArg,
                                    instruction.argsFormat.split(','),instruction.argsCount):

                    if fromPool:
                        arg = str(parser.constValue[arg-1])
                    else:
                        if 'switch' in instruction.mnemonic:
                            arg = frm.format(*arg)
                        else:
                            arg = frm.format(arg)
                    methodLines[str(byteIndex+offset) + " " + arg] = (len(embendingType)-1,embendingType[-1])
                    offset += size
                
            #print(methodLines[str(byteIndex) + " " + bytecode])
        
        methodLines[str(byteIndex) + " " + bytecode] = (embendingLayer[-1], embendingType[-1]) # implicit return is usually pointed
                                                                           # to function closing bracket
        codeByLines[method.name] = methodLines

    file.close()
    return codeByLines

def main():
    parser = argparse.ArgumentParser(description="Script for binding java bytecode to defined "+
                                                "classes for machine learning processing.")
    #group = parser.add_mutually_exclusive_group()
    #group.add_argument("-v", "--verbose", action="store_true")
    #group.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-cp", "--classpath",nargs='?',default=sys.path[0], 
                        help="Classpath for compiling java code.")
    parser.add_argument("-f", "--files", nargs='*',
                        help="Java classes for which to generate output, "+
                        "by default generates for all classes in classpath")
    parser.add_argument("-d", "--directory", nargs='?', default=sys.path[0]+'/.class_files', 
                        help="Directory for compiled class files, use -r to auto remove. "+
                        "By default its generated in .class_files under current directory.")
    parser.add_argument("-r", "--remove", action="store_true")
    parser.add_argument("-o", "--output", nargs='?', default=sys.path[0]+'/output', 
                        help="Path to output directory. By default in output in current directory of the script.")
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
    if process.returncode != 0: 
        print("javac failed %d %s %s" % (process.returncode, output, error))
    output = output.decode('utf-8')

    class_files=glob(save_class_directory+"/**",recursive=True)
    for class_file in class_files:
        if class_file.endswith(".class"):
            jf_class_name = class_file[:-6].split(save_class_directory)[1][1:]
            jf_name = jf_class_name.split('$')[0]
            for java_file in java_files:
                if jf_name in java_file:
                    source = java_file
                    break
            
            code_by_lines = bindCode(class_file, source)

            filename = output_folder+'/'+jf_class_name+".txt"
            
            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise


            f=open(filename,'w')
            for methodName in code_by_lines.keys():
                f.write("Method: "+str(methodName)+"\n")
                for bytecode in code_by_lines[methodName].keys():
                    f.write("\t"+bytecode+" "+str(code_by_lines[methodName][bytecode]) +"\n" )
                    
                f.write("\n")
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