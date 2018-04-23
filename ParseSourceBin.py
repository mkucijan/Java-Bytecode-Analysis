import argparse
import errno
import json
import os
import shutil
import subprocess
import sys

from copy import copy
from glob import glob

import javalang

from JavaClassParser.Parser import Parser


'''
Method used on javalang parser nodes for extracting
source code position of the node from the children nodes.
'''
def get_position(node):
    pos = getattr(node, 'position')
    if not pos:
        for attr_name in node.attrs:
            attr = getattr(node, attr_name)
            if isinstance(attr, javalang.ast.Node):
                if pos is None:
                    pos = get_position(attr)
                else:
                    break
            if isinstance(attr, list):
                for inner_node in attr:
                    if isinstance(inner_node, javalang.ast.Node):
                        if pos is None:
                            pos = get_position(inner_node)
                        else:
                            break

    return pos

'''
Naive method for binding conditional expresions with with its bytecode.
It finds the keywords in the source and includes all bytecode generated on that line
and in the parentheses of the expression. It uses lineNumberTable attribute from the
binary file parsed with JavaClassParser.
First class defined by number 0 represents bytecode not belonging to any conditional
statement.
'''
def bind_code(class_file, source):
    parser = Parser(class_file)
    parser.parse()
    code_by_lines = dict()
    file = open(source)
    source = file.read()
    tree = javalang.parse.parse(source)

    bin_methods = [method for method in parser.methods
                  if (method.name != '<init>')
                  and (method.name!='<clinit>')
                  and (not '$' in method.name)
                  and (getattr(method, 'code',None)!=None)]
    if not bin_methods:
        return code_by_lines
    source_methods = None
    if  '$' in class_file:
        inner_class_names = class_file[:-6].split('$')[1:]
        nodes = tree
        for inner_class_name in inner_class_names:
            if not inner_class_name.isdigit():
                decls = list(nodes.filter(javalang.tree.ClassDeclaration))
                flag_pass=0
                for node in decls:
                    if node[1].name == inner_class_name:
                        nodes = node[1]
                        flag_pass=1
                        break
                if not flag_pass:
                    decls = nodes.filter(javalang.tree.EnumDeclaration)
                    nodes = [node[1] for node in decls if node[1].name == inner_class_name][0]
                    nodes = nodes.body

            else:
                #if isinstance(nodes, javalang.tree.EnumBody):
                new_nodes = list(nodes.filter(javalang.tree.EnumConstantDeclaration))
                nodes = list(nodes.filter(javalang.tree.ClassCreator)) + new_nodes
                #else:
                #    nodes = list(nodes.filter(javalang.tree.ClassCreator))
                bin_method_names = [binMet.name for binMet in bin_methods]
                bin_method_pos = [binMet.code.lineNumberTable.code2Line[0] for binMet in bin_methods]
                nodeList = []
                numMatched = []
                diff_sums = []
                for node in nodes:
                    methods = []
                    if not node[1].body is None:
                        methods = [b[1] for b in node[1].filter(javalang.tree.MethodDeclaration)]
                    count = 0
                    positions = [get_position(m)[0] for m in methods]
                    diff_sum = 0
                    for i, m in enumerate(methods):
                        flag = 0
                        min_diff = positions[i]
                        for name, pos in zip(bin_method_names, bin_method_pos):
                            diff = abs(pos-positions[i])
                            if (m.name == name) and (min_diff>diff) :
                                diff_sum += diff
                                min_diff=diff
                                flag = 1
                        if flag:
                            count += 1
                    if count <= len(bin_method_names) and count>0:
                        nodeList.append(node[1])
                        numMatched.append(count)
                        diff_sums.append(diff_sum)
                maxMatch = 0
                if nodeList:
                    diff_min = max(diff_sums)
                    for numM, node, diff_sum in zip(numMatched, nodeList, diff_sums):
                        if (numM > maxMatch) or ((numM==maxMatch) and (diff_sum<diff_min)):
                            diff_min = diff_sum
                            maxMatch = numM
                            nodes = node
                #nodes = nodes[2*(int(inner_class_name)-1)+1][1]
        source_methods = []
        for nodeList in nodes.children:
            if isinstance(nodeList, list):
                for node in nodeList:
                    if isinstance(node, javalang.tree.MethodDeclaration):
                        source_methods.append(node)
    else:
        source_methods = getattr(tree, "types")[0].methods

    pom_bin_methods = bin_methods
    bin_methods = []
    for source_method in list(source_methods):
        new_method = None
        for j, bin_method in enumerate(pom_bin_methods):
            if bin_method.name==source_method.name:
                new_method = pom_bin_methods.pop(j)
                break
        if new_method:
            bin_methods.append(new_method)
        else:
            source_methods.remove(source_method)

    for binMethod, srcMethod in zip(bin_methods, source_methods):
        try:
            getattr(binMethod, 'code')
        except AttributeError:
            if 'abstract' not in str(binMethod):
                print(binMethod)
            continue
        methodLines = dict()
        instructions = binMethod.code.instructions
        lineNumberTable = binMethod.code.lineNumberTable.code2Line
        if srcMethod.name != binMethod.name:
            print("NAMES:")
            print(srcMethod.name)
            print(binMethod.name)
            print()
        statementInstructions = dict()
        statementName = []
        positions = []
        if srcMethod.body is None:
            continue
        for statement in srcMethod.body:
            # TODO Need to handle this situation
            pos = get_position(statement)
            if pos:
                statementName.append(type(statement).__name__)
                positions.append(pos[0])
        byteindexes = [instruction.startIndex for instruction in instructions]
        currentStatementIndex = -1
        positions.append(max(lineNumberTable.values()) + 1)
        if len(statementName) == 0:
                continue
        for instruction in instructions:
            byteIndex = instruction.startIndex
            if byteIndex in lineNumberTable:
                lineNumber = lineNumberTable[byteIndex]
                if lineNumber >= positions[currentStatementIndex+1]:
                    currentStatementIndex += 1
                    statementInstructions[statementName[currentStatementIndex]]=[]
                # TODO Something weird, apperntly not all final nodes hold position javalang problem
                if currentStatementIndex == -1:
                    break


            bytecode = instruction.mnemonic
            statementInstructions[statementName[currentStatementIndex]].append(
                (str(byteIndex), str(bytecode)))

            offset = 1
            if 'wide' in instruction.mnemonic:
                statementInstructions[statementName[currentStatementIndex]].append(
                    (str(byteIndex+offset),instruction.opcode.getMnemonic()) )
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
                    statementInstructions[statementName[currentStatementIndex]].append(
                        (str(byteIndex+offset),arg))
                    offset += size
        
        code_by_lines[binMethod.name] = statementInstructions
    
    file.close()
    return code_by_lines
        
        

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
            source = None
            flag = 1
            max_matches=0
            most_similar = None
            for java_file in java_files:
                matches = sum(a==b for a, b in zip(jf_name, java_file))
                if matches>max_matches:
                    max_matches=matches
                    most_similar = java_file
                if jf_name+".java" in java_file:
                    source = java_file
                    flag=0
                    break
            if flag:
                class_name = jf_name.split('/')[-1]
                if class_name in open(most_similar).read():
                    source = most_similar

            if source:
                code_by_lines = bind_code(class_file, source)
            else:
                continue

            filename = output_folder+'/'+jf_class_name+".json"

            if not os.path.exists(os.path.dirname(filename)):
                try:
                    os.makedirs(os.path.dirname(filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise


            f=open(filename,'w')
            json.dump(code_by_lines, f, indent=2)
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