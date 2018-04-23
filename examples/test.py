import javalang

source_file = open(
    "/home/yolkin/Documents/Project/Java-Bytecode-Analysis/examples/Test.java"
    ).read()
tree = javalang.parse.parse(source_file)
tokens = list(javalang.tokenizer.tokenize(source_file))
parser = javalang.parser.Parser(tokens)
tree2 = parser.parse()
#print(tree.types)

for token in tokens:
    print(token,token.position)
for path, node in tree.filter(javalang.tree.MethodDeclaration):
    print(node,node.position)
    print(type(node))

def get_pos(node):
    pos = getattr(node,'position')
    if not pos:
        for attrs in node.attrs:
            attr=getattr(node, attrs)
            if isinstance(attr,javalang.ast.Node):
                if attr is not None:
                    if pos is None:
                        pos = get_pos(attr)
                    else:
                        break    
    return pos


clasdecl=tree.types[0]
methods = clasdecl.methods
for method in methods:
    for b in method.body:
        print(type(b))
        print(get_pos(b))

print(tree)