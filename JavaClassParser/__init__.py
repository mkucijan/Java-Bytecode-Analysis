#from https://github.com/bloodlee/PyJavap PyJavapInternal
__author__ = 'jasonlee'

import struct

'''
from .Attribute import Attribute
from .ConstantPool import ConstantPool
from .Field import Field
from .Method import Method
from .Parser import Parser
from .ParsingException import ParsingException
from .ParsingResult import ParsingResult
'''

def ByteToHex(byteStr):
    """
    Convert a byte string to it's hex string representation e.g. for output.
    """

    # Uses list comprehension which is a fractionally faster implementation than
    # the alternative, more readable, implementation below
    #
    #    hex = []
    #    for aChar in byteStr:
    #        hex.append( "%02X " % ord( aChar ) )
    #
    #    return ''.join( hex ).strip()
    return ''.join('{:02x}'.format(x) for x in byteStr)

def HexToByte(hexStr):
    """
    Convert a string hex byte values into a byte string. The Hex Byte values may
    or may not be space separated.
    """
    # The list comprehension implementation is fractionally slower in this case
    #
    #    hexStr = ''.join( hexStr.split(" ") )
    #    return ''.join( ["%c" % chr( int ( hexStr[i:i+2],16 ) ) \
    #                                   for i in range(0, len( hexStr ), 2) ] )

    bytes = []

    hexStr = ''.join( hexStr.split(" ") )

    for i in range(0, len(hexStr), 2):
        bytes.append( chr( int (hexStr[i:i+2], 16 ) ) )

    return ''.join( bytes )

def ByteToDec(byteStr):
    return int(ByteToHex(byteStr), 16)

def ByteTo32BitFloat(_4bytes):
    return struct.unpack('>f', _4bytes)[0]

def ByteTo64BitFloat(_8bytes):
    return struct.unpack('>d', _8bytes)[0]

def doFlagToStr(flag, aDict):
    """
    Return the string of flag
    """

    result = ''

    for key in aDict.keys():
        if flag & key:
            result += aDict[key] + " "

    return result.strip()

