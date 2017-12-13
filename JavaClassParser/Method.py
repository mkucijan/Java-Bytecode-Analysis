__author__ = 'jason'


from JavaClassParser import MethodAccessFlags
from JavaClassParser import Attribute

class Method(object):

    def __init__(self, name, descriptor, access):
        self.name = name.decode('utf-8')
        self.descriptor = descriptor.decode('utf-8')
        self.access = access
        self.attrCount = 0
        self.attributes = []

    def addAttribute(self, attribute):
        self.attrCount += 1
        self.attributes.append(attribute)
        if attribute.name is Attribute.CODE_NAME:
            self.code = attribute

    def __str__(self):
        result = "Name: " + self.name + "(" + self.descriptor + ")"
        result += " Access: " + MethodAccessFlags.flagToStr(self.access)

        result += "\n"

        if len(self.attributes):
            result += "\tAttributes\n"
            for attr in self.attributes:
                result += "\t" + str(attr) + "\n"

        return result