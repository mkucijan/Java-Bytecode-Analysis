
import os.path
import os
from JavaClassParser.ParsingResult import ParsingResult
from JavaClassParser.ParsingException import ParsingException
from JavaClassParser import ByteToDec
from JavaClassParser.ConstantPool import ConstantPool
from JavaClassParser.Field import Field
from JavaClassParser.Method import Method
from JavaClassParser.Attribute import Attribute
from JavaClassParser.Attribute import *
from JavaClassParser.ExceptionInfo import ExceptionInfo

class Parser:


    def __init__(self, cls_file_name):
        self.cls_file_name = cls_file_name
        self.result = ParsingResult(self.cls_file_name)
        self.clsFile = None

    def __verify(self):
        """
        Verify the given arguments.
        """

        if not (os.path.exists(self.cls_file_name) and os.path.isfile(self.cls_file_name)):
            raise ParsingException("Can't find the class file %s" % self.cls_file_name)

    def __parseMagicNum(self):
        """
        Parse the magic num (the first 4 bytes)
        """

        assert self.clsFile is not None

        # get first four bytes
        magicNumBytes = self.clsFile.read(4)
        self.result.setMagicNumber(magicNumBytes)
        return magicNumBytes

    def __parseVersion(self):
        """
        Parse the version of this class file.
        Version is composed by minor version and major version.
        Each will occupy 2 bytes.
        """

        assert self.clsFile is not None

        minorVerBytes = self.clsFile.read(2)
        majorVerBytes = self.clsFile.read(2)

        self.result.setVersion(ByteToDec(minorVerBytes), ByteToDec(majorVerBytes))
        return minorVerBytes, majorVerBytes

    def __parseConstantPool(self):
        """
        Parse the constant pool of class file
        First 2 bytes of this section is the count.
        Pay attention here. Constant index starts from 1, not 0.
        """

        assert self.clsFile is not None

        const_count = ByteToDec(self.clsFile.read(2)) - 1
        self.result.setConstPoolCount(const_count)

        constants = ConstantPool.parse(const_count, self.clsFile)
        self.result.setConstants(constants)
        return constants


    def __parseAccessFlag(self):
        """
        Parse the access flag
        """

        assert self.clsFile is not None

        accessFlag = ByteToDec(self.clsFile.read(2))
        self.result.setAccessFlag(accessFlag)
        return accessFlag

    def __parseThis(self):
        """
        Parse "this" section
        """

        assert self.clsFile is not None

        thisIndex = ByteToDec(self.clsFile.read(2))
        self.result.setThisIndex(thisIndex)
        return thisIndex

    def __parseSuper(self):
        """
        Parse "super" section
        """

        assert self.clsFile is not None

        superIndex = ByteToDec(self.clsFile.read(2))
        self.result.setSuperIndex(superIndex)
        return superIndex

    def __parseInterface(self):
        """
        Parse "interface" section
        """

        assert self.clsFile is not None

        interfaceCount = ByteToDec(self.clsFile.read(2))

        interfaceIndex = []
        if interfaceCount > 0:
            for i in range(interfaceCount):
                interfaceIndex.append(ByteToDec(self.clsFile.read(2)))

        self.result.setInterfaces(interfaceCount, interfaceIndex)
        return interfaceIndex

    def __parseFields(self):
        """
        Parse fields sections.
        """

        assert self.clsFile is not None

        fieldCount = ByteToDec(self.clsFile.read(2))

        fields = []
        for index in range(fieldCount):
            accessFlag = ByteToDec(self.clsFile.read(2))
            name = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))
            descriptor = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))

            field = Field(name, descriptor, accessFlag)

            attrCount = ByteToDec(self.clsFile.read(2))
            if attrCount > 0:
                for i in range(attrCount):
                    attributeName = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))
                    attributeLength = ByteToDec(self.clsFile.read(4))

                    # for now, only parse the "Code Attribute
                    parser = Attribute.getParser(attributeName)

                    if parser is not None:
                         attribute = parser(self.clsFile, self.result)
                         field.addAttribute(attribute)
                    else:
                         self.clsFile.read(attributeLength)

            fields.append(field)

        self.result.setFields(fieldCount, fields)
        return fields

    def __parseMethods(self):
        """
        Parse methods section.
        """

        assert  self.clsFile is not None

        methodCount = ByteToDec(self.clsFile.read(2))

        methods = []
        for index in range(methodCount):
            accessFlag = ByteToDec(self.clsFile.read(2))
            name = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))
            descriptor = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))

            method = Method(name, descriptor, accessFlag)

            attrCount = ByteToDec(self.clsFile.read(2))
            if attrCount > 0:
                for i in range(attrCount):
                    attributeName = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))
                    attributeLength = ByteToDec(self.clsFile.read(4))
                    # for now, only parse the "Code Attribute
                    parser = Attribute.getParser(attributeName)

                    if parser is not None:
                        attribute = parser(self.clsFile, self.result)
                        method.addAttribute(attribute)
                    else:
                        self.clsFile.read(attributeLength)

            methods.append(method)

        self.result.setMethods(methodCount, methods)
        return methods

    def __parseAttribute(self):
        """
        Parse the attributes of class file.
        """

        attrCount = ByteToDec(self.clsFile.read(2))
        if attrCount > 0:
            for i in range(attrCount):
                attributeName = self.result.getUtf8(ByteToDec(self.clsFile.read(2)))
                attributeLength = ByteToDec(self.clsFile.read(4))

                parser = Attribute.getParser(attributeName)

                if parser is not None:
                    attribute = parser(self.clsFile, self.result)
                    self.result.addAttribute(attribute)
                    return attribute
                else:
                    self.clsFile.read(attributeLength)


    def parse(self):

        try:
            self.__verify()

            # parse the magic number of class file
            self.clsFile = open(self.cls_file_name, 'rb')

            # following functions' order is very important
            self.magic=self.__parseMagicNum()
            self.version=self.__parseVersion()
            self.constants=self.__parseConstantPool()
            self.flags=self.__parseAccessFlag()
            self.this=self.__parseThis()
            self.super=self.__parseSuper()
            self.interfaces=self.__parseInterface()
            self.fields=self.__parseFields()
            self.methods=self.__parseMethods()
            self.attribute=self.__parseAttribute()

        except ParsingException as e:
            raise e

        finally:
            if self.clsFile:
                self.clsFile.close()
                self.clsFile = None

        #return self.result

    