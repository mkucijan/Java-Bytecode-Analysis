# Java-Bytecode-Analysis
Analyzing java bytecode using machine learning.

## Dependencies
Installed java development kit is needed for:
- javac
- javap

## Usage
- 2 modules for generating train data
- Run BindSourceToBin.py or ConnByteSource.py with -h to see options

## Description
- BindSourceToBin.py compiles given java source code and creates labels for each instruction in the bytecode depending its encapsulation and the type such as if or while statement
- Seperately ConnByteSource.py lists for each line of the source code generated bytecode instructions