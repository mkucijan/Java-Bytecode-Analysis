import plyj.parser 
import plyj.model as m

def main():
    parser = plyj.parser.Parser()
    tree = parser.parse_file('/home/yolkin/Documents/Project/Java-Bytecode-Analysis/examples/Test.java')
    print(tree)
    print()
    print(tree.type_declarations)
    methods = [[met_decl for met_decl in type_decl.body if type(met_decl) is m.MethodDeclaration] for type_decl in tree.type_declarations ]
    print()
    print()
    for meth in methods:
        for meth_decl in meth:
            for line in meth_decl.body:
                print(line)
        
        #print(meth)
        print()
if __name__ == '__main__':
    main()
    