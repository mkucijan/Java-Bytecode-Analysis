from JavaClassParser.Parser import Parser

if __name__ == '__main__':
    parser = Parser('/home/yolkin/Documents/Project/Java-Bytecode-Analysis/examples/Test.class')
    parser.parse()
    #print(parser.magic)
    #print(parser.version)
    #for i,constant in zip(range(1,len(parser.constants)+1),parser.constants):
    #    print("#"+str(i),str(constant)+"\n")
    #print(parser.this)
    #print(parser.super)
    #print(parser.interfaces)
    #print(parser.fields)
    print([method.attrCount for method in parser.methods])
    for method in parser.methods:
        print(str(method))
        #for attribute in method.attributes:
            #print(attribute)
        
    instructions = parser.methods[-1].code.instructions
    for instruction in instructions:
        print(instruction.bytecode)

    for const in parser.constants:
        print(str(const))

    i=0
    for const in parser.constValue:
        i+=1
        print("#",i,const)