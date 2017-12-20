__author__ = 'yli'

from copy import deepcopy
import struct
        

class ByteCode(object):

    def __init__(self, byteCode, mnemonic, opCodeCount=0,args=''):
        self.byteCode = byteCode
        self.mnemonic = mnemonic
        self.opCodeCount = opCodeCount
        self.parsedString = None
        self.argsCount=[]
        self.argsFormat=''
        self.args = args
        if args:
            for arg in args.split(','):
                if '0' in arg:
                    break
                if '#' in arg:
                    arg = arg.replace('#','')
                    arg = arg.replace('c','')
                    self.argsFormat += '#{:d} '
                    self.argsCount.append(int(arg))
                else:
                    self.argsFormat += '{:d} '
                    self.argsCount.append(int(arg))
            self.argsFormat = self.argsFormat[:-1]
            self.argsFormat = self.argsFormat.replace(' ',',')

    def getByteCode(self):
        return self.byteCode

    def getMnemonic(self):
        return self.mnemonic

    def getOpCodeCount(self):
        return self.opCodeCount

    def __eq__(self, other):
        return self.byteCode == other.byteCode

    def parse(self, code, byteIndex):
        result = '\t\t%d: %s ' % (byteIndex / 2, self.getMnemonic())
        values = []
        byteIndex += 2
        self.constArg = []
        for argSize, frm in zip(self.argsCount,self.args.split(',')):
            self.constArg.append(False)
            if argSize:
                if '#' in frm:
                    values.append(ByteCode.getBigEndianInt(code[byteIndex:byteIndex+argSize*2]))
                    if 'c' in frm:
                        self.constArg[-1] = True
                else:
                    values.append(ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+argSize*2]))
                byteIndex += argSize*2
            byteIndex += 2
        if self.argsCount:
            result += self.argsFormat.format(*values)
        self.argValues = values
        result += '\n'
        self.parsedString = result
        return result

    def __call__(self,code,byteIndex):
        self.startIndex = byteIndex//2
        self.parse(code,byteIndex)
        self.bytecode = code[byteIndex:byteIndex+2+self.opCodeCount*2]
        return deepcopy(self)

    def __hash__(self):
        return self.bytecode.__hash__
    
    def __str__(self):
        if self.parsedString:
            return self.parsedString
        return "%s(0x%x)" % (self.mnemonic, self.byteCode)

    '''
        Get value in big endian order.
    '''
    @staticmethod
    def getBigEndianInt(operandBytes):
        return int(operandBytes,16)

    '''
        Get big endian signed.
    '''
    @staticmethod
    def getBigEndianSinged(operandBytes):
        masklen=len(operandBytes)
        mask = 'F'*(masklen-1)
        mask = '7'+mask
        value = int(operandBytes,16)
        mask = int(mask,16)
        if (value > mask):
            value -= int('1'+'0'*masklen,16)
        return value
    
class LookupSwitchBytecode(ByteCode):
    
    def __init__(self):
        super().__init__(0xab, 'lookupswitch',-1)
        self.parsedString = self.mnemonic

    def parse(self, code, byteIndex):
        result = '\t\t%d: %s  {\n' % (byteIndex / 2, self.getMnemonic())
        startByteIndex = byteIndex
        padding = int( byteIndex / 2 + 1 ) % 4
        padding = (4-padding) % 4
        byteIndex += (1+padding)*2
        default_offset = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8]) + \
                         int(startByteIndex/2)
        byteIndex += 8
        npairs = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8])
        byteIndex += 8
        self.args=''
        self.argsCount=[]
        self.argsFormat=''
        self.argValues=[]
        self.constArg=[]
        for i in range(npairs):
            value = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8]) 
            byteIndex += 8
            offset = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8]) + \
                     int(startByteIndex/2)
            byteIndex += 8
            case = "\t\t\t%7d: %d\n" % (value,offset)
            result += case

            self.args += '4,'
            self.argsCount.append(4)
            self.argsFormat += '{:d}: {:d},'
            self.argValues.append((value,offset))
            self.constArg.append(False)
        
        self.args = self.args[:-1]
        self.argsFormat = self.argsFormat[:-1]

        result += "\t\t\tdefault: %d\n\t\t}\n" % (default_offset)
        self.opCodeCount = int((byteIndex-startByteIndex)/2 - 1)
        self.parsedString = result

        return result 


class TableSwitchBytecode(ByteCode):
    
    def __init__(self):
        super().__init__(0xaa, 'tableswitch', -1)

    def parse(self, code, byteIndex):
        result = '\t\t%d: %s  {\n' % (byteIndex / 2, self.getMnemonic())
        startByteIndex = byteIndex
        padding = int( byteIndex / 2 + 1 ) % 4
        padding = (4-padding) % 4
        byteIndex += (1+padding)*2
        default_offset = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8]) + \
                         int(startByteIndex/2)
        byteIndex += 8
        lowbyte = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8])
        byteIndex += 8
        highbyte = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8])
        byteIndex += 8
        N = highbyte - lowbyte + 1 
        
        self.args=''
        self.argsCount=[]
        self.argsFormat=''
        self.argValues=[]
        self.constArg=[]
        
        for i in range(N):
            offset = ByteCode.getBigEndianSinged(code[byteIndex:byteIndex+8]) + \
                     int(startByteIndex/2)
            byteIndex += 8
            case = "\t\t\t%7d: %d\n" % (lowbyte+i,offset)
            result += case

            self.args += '4,'
            self.argsCount.append(4)
            self.argsFormat += '{:d} {:d},'
            self.argValues.append((lowbyte+i, offset))
            self.constArg.append(False)
        
        self.args = self.args[:-1]
        self.argsFormat = self.argsFormat[:-1]

        result += "\t\t\tdefault: %d\n\t\t}\n" % (default_offset)
        self.opCodeCount = int((byteIndex-startByteIndex)/2 - 1)
        self.parsedString = result
        return result 
    

class WideByteCode(ByteCode):
    
    def __init__(self):
        super().__init__(0xc4, 'wide', -1)
        self.parsedString = self.mnemonic
    
    def parse(self, code, byteIndex):
        result = '\t\t%d: %s  ' % (byteIndex / 2, self.getMnemonic())
        startByteIndex = byteIndex
        byteIndex += 2
        
        self.opcode = codeMap[ByteCode.getBigEndianInt(code[byteIndex:byteIndex+2])]
  
        result += '%s ' % (self.opcode.getMnemonic())

        index = ByteCode.getBigEndianInt(code[byteIndex:byteIndex+step])
        byteIndex += 2

        result += '#%d ' % (index)


        self.args='#2'
        self.argsCount=[2]
        self.argsFormat='{:d}'
        self.argValues=[index]
        self.constArg=[False]

        if opcode is 'iinc':
            count = ByteCode.getBigEndianInt(code[byteIndex:byteIndex+step])
            byteIndex += 2
            result += ',%d' % (count)
            self.args += ',2'
            self.argsCount.append(2)
            self.argsFormat += ',{:d}'
            self.argValues.append(count)
            self.constArg.append(False)

        result += '\n'        
        self.opCodeCount = int((byteIndex-startByteIndex)/2 - 1)
        self.parsedString = result
        return result


allByteCodes = [
    ByteCode(0x32, 'aaload'),
    ByteCode(0x53, 'aastore'),
    ByteCode(0x01, 'aconst_null'),
    ByteCode(0x19, 'aload', 1,args='#1'),
    ByteCode(0x2a, 'aload_0'),
    ByteCode(0x2b, 'aload_1'),
    ByteCode(0x2c, 'aload_2'),
    ByteCode(0x2d, 'aload_3'),
    ByteCode(0xdb, 'anewarray', 2,args='c#2'),
    ByteCode(0xb0, 'areturn'),
    ByteCode(0xbe, 'arraylength'),
    ByteCode(0x3a, 'astore', 1,args='#1'),
    ByteCode(0x4b, 'astore_0'),
    ByteCode(0x4c, 'astore_1'),
    ByteCode(0x4d, 'astore_2'),
    ByteCode(0x4e, 'astore_3'),
    ByteCode(0xbf, 'athrow'),

    ByteCode(0x33, 'baload'),
    ByteCode(0x54, 'bastore'),
    ByteCode(0x10, 'bipush',1,args='1'),

    ByteCode(0x34, 'caload'),
    ByteCode(0x55, 'castore'),
    ByteCode(0xc0, 'checkcast', 2,args='c#2'),

    ByteCode(0x90, 'd2f'),
    ByteCode(0x8e, 'd2i'),
    ByteCode(0x8f, 'd2l'),
    ByteCode(0x63, 'dadd'),
    ByteCode(0x31, 'daload'),
    ByteCode(0x52, 'dastore'),
    ByteCode(0x98, 'dcmpg'),
    ByteCode(0x97, 'dcmpl'),
    ByteCode(0xe,  'dconst_0'),
    ByteCode(0xf,  'dconst_1'),
    ByteCode(0x6f, 'ddiv'),
    ByteCode(0x18, 'dload', 1,args='#1'),
    ByteCode(0x26, 'dload_0'),
    ByteCode(0x27, 'dload_1'),
    ByteCode(0x28, 'dload_2'),
    ByteCode(0x29, 'dload_3'),
    ByteCode(0x6b, 'dmul'),
    ByteCode(0x77, 'dneg'),
    ByteCode(0x73, 'drem'),
    ByteCode(0xaf, 'dreturn'),
    ByteCode(0x39, 'dstore', 1,args='#1'),
    ByteCode(0x47, 'dstore_0'),
    ByteCode(0x48, 'dstore_1'),
    ByteCode(0x49, 'dstore_2'),
    ByteCode(0x4a, 'dstore_3'),
    ByteCode(0x67, 'dsub'),
    ByteCode(0x59, 'dup'),
    ByteCode(0x5a, 'dup_x1'),
    ByteCode(0x5b, 'dup_x2'),
    ByteCode(0x5c, 'dup2'),
    ByteCode(0x5d, 'dup2_x1'),
    ByteCode(0x5e, 'dup2_x2'),

    ByteCode(0x8d, 'f2d'),
    ByteCode(0x8b, 'f2i'),
    ByteCode(0x8c, 'f2l'),
    ByteCode(0x62, 'fadd'),
    ByteCode(0x30, 'faload'),
    ByteCode(0x51, 'fastore'),
    ByteCode(0x96, 'fcmpg'),
    ByteCode(0x95, 'fcmpl'),
    ByteCode(0xb,  'fconst_0'),
    ByteCode(0xc,  'fconst_1'),
    ByteCode(0xd,  'fconst_2'),
    ByteCode(0x6e, 'fdiv'),
    ByteCode(0x17, 'fload', 1,args='#1'),
    ByteCode(0x22, 'fload_0'),
    ByteCode(0x23, 'fload_1'),
    ByteCode(0x24, 'fload_2'),
    ByteCode(0x25, 'fload_3'),
    ByteCode(0x6a, 'fmul'),
    ByteCode(0x76, 'fneg'),
    ByteCode(0x72, 'frem'),
    ByteCode(0xae, 'freturn'),
    ByteCode(0x38, 'fstore', 1,args='#1'),
    ByteCode(0x43, 'fstore_0'),
    ByteCode(0x44, 'fstore_1'),
    ByteCode(0x45, 'fstore_2'),
    ByteCode(0x46, 'fstore_3'),
    ByteCode(0x66, 'fsub'),

    ByteCode(0xb4, 'getfield', 2,args='c#2'),
    ByteCode(0xb2, 'getstatic', 2,args='c#2'),
    ByteCode(0xa7, 'goto', 2,args='2'),
    ByteCode(0xc8, 'goto_w', 4,args='4'),

    ByteCode(0x91, 'i2b'),
    ByteCode(0x92, 'i2c'),
    ByteCode(0x87, 'i2d'),
    ByteCode(0x86, 'i2f'),
    ByteCode(0x85, 'i2l'),
    ByteCode(0x93, 'i2s'),
    ByteCode(0x60, 'iadd'),
    ByteCode(0x2e, 'iaload'),
    ByteCode(0x7e, 'iand'),
    ByteCode(0x4f, 'iastore'),
    ByteCode(0x2,  'iconst_m1'),
    ByteCode(0x3,  'iconst_0'),
    ByteCode(0x4,  'iconst_1'),
    ByteCode(0x5,  'iconst_2'),
    ByteCode(0x6,  'iconst_3'),
    ByteCode(0x7,  'iconst_4'),
    ByteCode(0x8,  'iconst_5'),
    ByteCode(0x6c, 'idiv'),
    ByteCode(0xa5, 'if_acmpeq', 2,args='2'),
    ByteCode(0xa6, 'if_acmpne', 2,args='2'),
    ByteCode(0x9f, 'if_icmpeq', 2,args='2'),
    ByteCode(0xa0, 'if_icmpne', 2,args='2'),
    ByteCode(0xa1, 'if_icmplt', 2,args='2'),
    ByteCode(0xa2, 'if_icmpge', 2,args='2'),
    ByteCode(0xa3, 'if_icmpgt', 2,args='2'),
    ByteCode(0xa4, 'if_icmple', 2,args='2'),
    ByteCode(0x99, 'ifeq', 2,args='2'),
    ByteCode(0x9a, 'ifne', 2,args='2'),
    ByteCode(0x9b, 'iflt', 2,args='2'),
    ByteCode(0x9c, 'ifge', 2,args='2'),
    ByteCode(0x9d, 'ifgt', 2,args='2'),
    ByteCode(0x9e, 'ifle', 2,args='2'),
    ByteCode(0xc7, 'ifnonnull', 2,args='2'),
    ByteCode(0xc6, 'ifnull', 2,args='2'),
    ByteCode(0x84, 'iinc', 2,args='#1,1'),
    ByteCode(0x15, 'iload', 1,args='#1'),
    ByteCode(0x1a, 'iload_0'),
    ByteCode(0x1b, 'iload_1'),
    ByteCode(0x1c, 'iload_2'),
    ByteCode(0x1d, 'iload_3'),
    ByteCode(0x68, 'imul'),
    ByteCode(0x74, 'ineg'),
    ByteCode(0xc1, 'instanceof', 2,args='c#2'),
    ByteCode(0xba, 'invokedynamic', 4,args='c#2,0,0'),
    ByteCode(0xb9, 'invokeinterface', 4,args='c#2,1,0'),
    ByteCode(0xb7, 'invokespecial', 2,args='c#2'),
    ByteCode(0xb8, 'invokestatic', 2,args='c#2'),
    ByteCode(0xb6, 'invokevirtual', 2,args='c#2'),
    ByteCode(0x80, 'ior'),
    ByteCode(0x70, 'irem'),
    ByteCode(0xac, 'ireturn'),
    ByteCode(0x78, 'ishl'),
    ByteCode(0x7a, 'ishr'),
    ByteCode(0x36, 'istore', 1,args='#1'),
    ByteCode(0x3b, 'istore_0'),
    ByteCode(0x3c, 'istore_1'),
    ByteCode(0x3d, 'istore_2'),
    ByteCode(0x3e, 'istore_3'),
    ByteCode(0x64, 'isub'),
    ByteCode(0x7c, 'iushr'),
    ByteCode(0x82, 'ixor'),

    ByteCode(0xa8, 'jsr', 2,args='2'),
    ByteCode(0xc9, 'jsr_w', 4,args='4'),

    ByteCode(0x8a, 'l2d'),
    ByteCode(0x89, 'l2f'),
    ByteCode(0x88, 'l2i'),
    ByteCode(0x61, 'ladd'),
    ByteCode(0x2f, 'laload'),
    ByteCode(0x7f, 'land'),
    ByteCode(0x50, 'lastore'),
    ByteCode(0x94, 'lcmp'),
    ByteCode(0x9,  'lconst_0'),
    ByteCode(0xa,  'lconst_1'),
    ByteCode(0x12, 'ldc', 1,args='c#1'),
    ByteCode(0x13, 'ldc_w', 2,args='c#2'),
    ByteCode(0x14, 'ldc2_w', 2,args='c#2'),
    ByteCode(0x6d, 'ldiv'),
    ByteCode(0x16, 'lload', 1,args='#1'),
    ByteCode(0x1e, 'lload_0'),
    ByteCode(0x1f, 'lload_1'),
    ByteCode(0x20, 'lload_2'),
    ByteCode(0x21, 'lload_3'),
    ByteCode(0x69, 'lmul'),
    ByteCode(0x75, 'lneg'),

    LookupSwitchBytecode(),

    ByteCode(0x81, 'lor'),
    ByteCode(0x71, 'lrem'),
    ByteCode(0xad, 'lreturn'),
    ByteCode(0x79, 'lshl'),
    ByteCode(0x7b, 'lshr'),
    ByteCode(0x37, 'lstore', 1,args='#1'),
    ByteCode(0x3f, 'lstore_0'),
    ByteCode(0x40, 'lstore_1'),
    ByteCode(0x41, 'lstore_2'),
    ByteCode(0x42, 'lstore_3'),
    ByteCode(0x65, 'lsub'),
    ByteCode(0x7d, 'lushr'),
    ByteCode(0x83, 'lxor'),

    ByteCode(0xc2, 'monitorenter'),
    ByteCode(0xc3, 'monitorexit'),
    ByteCode(0xc5, 'multianewarray', 3,args='c#2,1'),

    ByteCode(0xbb, 'new', 2,args='c#2'),
    ByteCode(0xbc, 'newarray', 1,args='1'),
    ByteCode(0xbd, 'anewarray',2,args='c#2'),
    ByteCode(0xca, 'breakpoint'),

    ByteCode(0x00, 'nop'),

    ByteCode(0x57, 'pop'),
    ByteCode(0x58, 'pop2'),
    ByteCode(0xb5, 'putfield', 2,args='c#2'),
    ByteCode(0xb3, 'putstatic', 2,args='c#2'),

    ByteCode(0xa9, 'ret', 1,args='#1'),
    ByteCode(0xb1, 'return'),

    ByteCode(0x35, 'saload'),
    ByteCode(0x56, 'sastore'),
    ByteCode(0x11, 'sipush', 2,args='2'),
    ByteCode(0x5f, 'swap'),

    TableSwitchBytecode(),

    WideByteCode(),

]

def getByteCodeMap():
    codeMap = dict()

    for aByteCode in allByteCodes:
        codeMap[aByteCode.getByteCode()] = aByteCode

    return codeMap

codeMap = getByteCodeMap()

def getMnemonicMap():
    mnemonicMap = dict()
    
    for aMnemonic in allByteCodes:
        mnemonicMap[aMnemonic.getMnemonic()] = aMnemonic
    
    return mnemonicMap

mnemonicMap = getMnemonicMap