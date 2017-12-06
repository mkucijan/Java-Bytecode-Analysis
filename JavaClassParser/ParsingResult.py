
import JavaClassParser
from JavaClassParser import ClassAccessFlags, FieldAccessFlags
from JavaClassParser.ConstantPool import ConstantClassInfo, ConstantUTF8Info
from JavaClassParser.ParsingException import ParsingException
from JavaClassParser.ConstantPool import *

class ParsingResult:

    def __init__(self, cls_file_name):
        self.cls_file_name = cls_file_name
        self.magicNumber = [0xff, 0xff, 0xff, 0xff]
        self.majorVer = -1
        self.minorVer = -1

        self.const_pool_count = -1
        self.constants = None

        self.accessFlag = 0x0000

        self.thisIndex = -1
        self.superIndex = -1

        self.interfaceCount = -1
        self.interfaces = None

        self.fieldCount = -1
        self.fields = None

        self.methodCount = -1
        self.methods = None

        self.attributes = []

    def setMagicNumber(self, magicNumber):
        self.magicNumber = magicNumber

    def getMagicNumber(self):
        return self.magicNumber

    def setVersion(self, major, minor):
        self.majorVer = major
        self.minorVer = minor

    def getVersions(self):
        return (self.majorVer, self.minorVer)

    def setConstPoolCount(self, count):
        self.const_pool_count = count

    def setConstants(self, constants):
        self.constants = constants

    def getConstants(self):
        return self.constants

    def setAccessFlag(self, accessFlag):
        self.accessFlag = accessFlag

    def getAccessFlag(self):
        return self.accessFlag

    def setThisIndex(self, thisIndex):
        self.thisIndex = thisIndex

    def setSuperIndex(self, superIndex):
        self.superIndex = superIndex

    def setInterfaces(self, count, interfaces):
        self.interfaceCount = count
        self.interfaces = interfaces

    def setFields(self, count, fields):
        self.fieldCount = count
        self.fields = fields

    def setMethods(self, count, methods):
        self.methodCount = count
        self.methods = methods

    def addAttribute(self, attribute):
        self.attributes.append(attribute)

    def __str__(self):

        result = ""

        result += "File Name: %s\n" % self.cls_file_name
        result += "Magic Number: %s\n" % JavaClassParser.ByteToHex(self.magicNumber)
        result += "Major Version: %d\n" % self.majorVer
        result += "Minor Version: %d\n" % self.minorVer

        result += "Constant Pool (Count %d)\n" % len(self.constants)

        if len(self.constants) > 0:
            if self.constants is not None:
                result += "==========================\n"

                for i in range(len(self.constants)):
                    result += 'const #%d = %s\n' % (i + 1, str(self.constants[i]))

            result += "==========================\n"

        result += "Access Flag: %s\n" % ClassAccessFlags.flagToStr(self.accessFlag)

        if self.thisIndex >= 1:
            result += "this: %s\n" % self.getClassName(self.thisIndex)

        if self.superIndex >= 1:
            result += "super: %s\n" % self.getClassName(self.superIndex)

        if len(self.attributes):
            result += "Attributes\n"
            result += "=====================\n"

            for attribute in self.attributes:
                result += str(attribute) + "\n"

            result += "\n"

        if self.interfaceCount > 0:

            result += "Implemented interfaces: "
            for i in range(self.interfaceCount):
                result += self.getClassName(self.interfaces[i]) + ","
            result = result[:-1] # eliminate the last ','
            result += "\n"

        else:
            result += "No interface\n"


        if self.fieldCount > 0:

            result += "Fields(%d): \n" % self.fieldCount
            for i in range(self.fieldCount):
                result += "\t" + str(self.fields[i]) + "\n"

        if self.methodCount > 0:

            result += "Methods(%d): \n" % self.methodCount
            for i in range(self.methodCount):
                result += "\t" + str(self.methods[i]) + "\n"

        return result

    def getClassName(self, index):
        """
        Get the class from constant pool.
        """

        if index in range(1, self.const_pool_count + 1):

            # get the const class info first
            classInfo = self.constants[index - 1]
            nameIndex = classInfo.getNameIndex()

            # get class name from UTF8 info
            className = self.constants[nameIndex - 1].getUtf8()

            return className
        else:
            raise ParsingException('Const index is out of range.')

    def getUtf8(self, index):

        if index in range(1, self.const_pool_count + 1):

            typeName = self.constants[index - 1].getTypeName()
            if typeName == TYPE_UTF8:
                return self.constants[index - 1].getUtf8()
            else:
                raise ParsingException('The index is not for UTF8 info.')
        else:
            raise ParsingException('Index is out of range.')
