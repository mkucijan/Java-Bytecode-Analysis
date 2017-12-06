import sys
from collections import namedtuple
import struct
import binascii
import re

IMAGE_DOS_HEADER_fields=(
'Magic',
'BytesOnLastPageOfFile',
'PagesInFile',
'Relocations',
'SizeOfHeaderInParagraphs',
'MinimumExtraParagraphs',
'MaximumExtraParagraphs',
'Initial_relative_SS',
'InitialSP',
'Checksum',
'InitialIP',
'Initial_relative_CS',
'OffsetToRelocationTable',
'OverlayNumber',
'Reserved1',
'Reserved2',
'Reserved3',
'Reserved4',
'OEMIndentifier',
'OEMInformation',
'Reserved5',
'Reserved6',
'Reserved7',
'Reserved8',
'Reserved9',
'Reserved10',
'Reserved11',
'Reserved12',
'Reserved13',
'Reserved14',
'OffsetToNewExeHeader'
)

IMAGE_DOS_HEADER_format=[2]*30+[4]
IMAGE_DOS_HEADER_size=64
IMAGE_DOS_HEADER_struct=namedtuple('IMAGE_DOS_HEADER',' '.join(IMAGE_DOS_HEADER_fields))

IMAGE_FILE_HEADER_fields=(
'Magic',
'Machine',
'NumberOfSections',
'TimeDateStamp',
'PointerToSymbolTable',
'NumberOfSymbols',
'SizeOfOptionalHeader',
'Characteristics'
)
IMAGE_FILE_HEADER_format=[4]+[2]*2+[4]*3+[2]*2
IMAGE_FILE_HEADER_size=24
IMAGE_FILE_HEADER_struct=namedtuple('IMAGE_FILE_HEADER',' '.join(IMAGE_FILE_HEADER_fields))

IMAGE_OPTIONAL_HEADER_fields=(
'Magic',
'MajorLinkerVersion',
'MinorLinkerVersion',
'SizeOfCode',
'SizeOfInitializedData',
'SizeOfUninitializedData',
'AddressOfEntryPoint',
'BaseOfCode',
'BaseOfData',
'ImageBase',
'SectionAlignment',
'FileAlignment',
'MajorOperatingSystemVersion',
'MinorOperatingSystemVersion',
'MajorImageVersion',
'MinorImageVersion',
'MajorSubsystemVersion',
'MinorSubsystemVersion',
'Win32VersionValue',
'SizeOfImage',
'SizeOfHeaders',
'CheckSum',
'Subsystem',
'DllCharacteristics',
'SizeOfStackReserve',
'SizeOfStackCommit',
'SizeOfHeapReserve',
'SizeOfHeapCommit',
'LoaderFlags',
'NumberOfDataDirectories'
)
IMAGE_OPTIONAL_HEADER_format=[2]+[1]*2+[4]*9+[2]*6+[4]*4+[2]*2+[4]*6
IMAGE_OPTIONAL_HEADER_size=96
IMAGE_OPTIONAL_HEADER_struct=namedtuple('IMAGE_OPTIONAL_HEADER',' '.join(IMAGE_OPTIONAL_HEADER_fields))

IMAGE_DATA_DIRECTORY_fields=(
'VirtualAddress',
'Size'
)
IMAGE_DATA_DIRECTORY_format=[4]*2
IMAGE_DATA_DIRECTORY_size=8
IMAGE_DATA_DIRECTORY_struct=namedtuple('IMAGE_DATA_DIRECTORY',' '.join(IMAGE_DATA_DIRECTORY_fields))

IMAGE_DATA_DIRECTORIES_descrition=(
'Export table',
'Import table',
'Resource table',
'Exception table',
'Certificate table',
'Base relocation table',
'Debugging information',
'Architecture-specific',
'Global pointer register',
'Thread local storage (TLS)',
'Load configuration table',
'Bound import table',
'Import address table',
'Delay import descriptor',
'The CLR header',
'Reserved'
)

IMAGE_SECTION_HEADER_fields=(
'Name',
'VirtualSize',
'VirtualAddress',
'SizeOfRawData',
'PointerToRawData',
'PointerToRelocations',
'PointerToLinenumbers',
'NumberOfRelocations',
'NumberOfLinenumbers',
'Characteristics',
)

IMAGE_SECTION_HEADER_format=[8]+[4]*6+[2]*2+[4]
IMAGE_SECTION_HEADER_size=40
IMAGE_SECTION_HEADER_struct=namedtuple('IMAGE_SECTION_HEADER',' '.join(IMAGE_SECTION_HEADER_fields))

IMPORT_DIRECTORY_fields=(
'ImportNameTableRVA',
'TimeDateStamp',
'ForwarderChain',
'NameRVA',
'ImportAddressTableRVA'
)

IMPORT_DIRECTORY_format=[4]*5
IMPORT_DIRECTORY_size=20
IMPORT_DIRECTORY_struct=namedtuple('IMPORT_DIRECTORY',' '.join(IMPORT_DIRECTORY_fields))

IMAGE_EXPORT_DIRECTORY_fields=(
'Characteristics',
'TimeDateStamp',
'MajorVersion',
'MinorVersion',
'NameRVA',
'OrdinalBase',
'NumberOfFunctions',
'NumberOfNames',
'AddressTableRVA',
'NamePointerTableRVA',
'OrdinalTableRVA'
)

IMAGE_EXPORT_DIRECTORY_format=[4]*2+[2]*2+[4]*7
IMAGE_EXPORT_DIRECTORY_size=40
IMAGE_EXPORT_DIRECTORY_struct=namedtuple('IMAGE_EXPORT_DIRECTORY',' '.join(IMAGE_EXPORT_DIRECTORY_fields))

def main(argv1):
    pe_file=open(argv1,'rb').read()

    SectionAdrStart=0
    SectionAdrEnd=IMAGE_DOS_HEADER_size
    IMAGE_DOS_HEADER_values=struct.unpack(format_for_unpack(IMAGE_DOS_HEADER_format),pe_file[SectionAdrStart:SectionAdrEnd])
    IMAGE_DOS_HEADER=IMAGE_DOS_HEADER_struct._make(IMAGE_DOS_HEADER_values)
    print_section(IMAGE_DOS_HEADER,IMAGE_DOS_HEADER_format)

    SectionAdrStart=IMAGE_DOS_HEADER.OffsetToNewExeHeader
    SectionAdrEnd=IMAGE_DOS_HEADER.OffsetToNewExeHeader+IMAGE_FILE_HEADER_size
    IMAGE_FILE_HEADER_values=struct.unpack(format_for_unpack(IMAGE_FILE_HEADER_format),
        pe_file[SectionAdrStart:SectionAdrEnd])
    IMAGE_FILE_HEADER=IMAGE_FILE_HEADER_struct._make(IMAGE_FILE_HEADER_values)
    print_section(IMAGE_FILE_HEADER,IMAGE_FILE_HEADER_format)

    SectionAdrStart=SectionAdrEnd
    SectionAdrEnd=SectionAdrStart+IMAGE_OPTIONAL_HEADER_size
    IMAGE_OPTIONAL_HEADER_values=struct.unpack(format_for_unpack(IMAGE_OPTIONAL_HEADER_format),
        pe_file[SectionAdrStart:SectionAdrEnd])
    IMAGE_OPTIONAL_HEADER=IMAGE_OPTIONAL_HEADER_struct._make(IMAGE_OPTIONAL_HEADER_values)
    print_section(IMAGE_OPTIONAL_HEADER,IMAGE_OPTIONAL_HEADER_format)

    IMAGE_DATA_DIRECTORIES=[]
    for i in range(IMAGE_OPTIONAL_HEADER.NumberOfDataDirectories):
        SectionAdrStart=SectionAdrEnd
        SectionAdrEnd=SectionAdrStart+IMAGE_DATA_DIRECTORY_size
        IMAGE_DATA_DIRECTORY_values=struct.unpack(format_for_unpack(IMAGE_DATA_DIRECTORY_format),
            pe_file[SectionAdrStart:SectionAdrEnd])
        IMAGE_DATA_DIRECTORY=IMAGE_DATA_DIRECTORY_struct._make(IMAGE_DATA_DIRECTORY_values)
        print_section(IMAGE_DATA_DIRECTORY,IMAGE_DATA_DIRECTORY_format,IMAGE_DATA_DIRECTORIES_descrition[i])
        IMAGE_DATA_DIRECTORIES.append(IMAGE_DATA_DIRECTORY)

    IMAGE_SECTION_HEADERs=[]
    for i in range(IMAGE_FILE_HEADER.NumberOfSections):
        SectionAdrStart=SectionAdrEnd
        SectionAdrEnd=SectionAdrStart+IMAGE_SECTION_HEADER_size
        IMAGE_SECTION_HEADER_values=struct.unpack(format_for_unpack(IMAGE_SECTION_HEADER_format),
            pe_file[SectionAdrStart:SectionAdrEnd])
        IMAGE_SECTION_HEADER=IMAGE_SECTION_HEADER_struct._make(IMAGE_SECTION_HEADER_values)
        print_section(IMAGE_SECTION_HEADER,IMAGE_SECTION_HEADER_format)
        IMAGE_SECTION_HEADERs.append(IMAGE_SECTION_HEADER)
        if ((IMAGE_SECTION_HEADER.VirtualAddress<=IMAGE_DATA_DIRECTORIES[1].VirtualAddress) and \
            ((IMAGE_SECTION_HEADER.VirtualAddress+IMAGE_OPTIONAL_HEADER.SectionAlignment)>IMAGE_DATA_DIRECTORIES[1].VirtualAddress)):
            Import_table_adr=(IMAGE_DATA_DIRECTORIES[1].VirtualAddress-IMAGE_SECTION_HEADER.VirtualAddress) + \
                IMAGE_SECTION_HEADER.PointerToRawData
            IMPORT_SECTION_HEADER=IMAGE_SECTION_HEADER
        if (IMAGE_DATA_DIRECTORIES[0].VirtualAddress!=0):
            if ((IMAGE_SECTION_HEADER.VirtualAddress<=IMAGE_DATA_DIRECTORIES[0].VirtualAddress) and \
                ((IMAGE_SECTION_HEADER.VirtualAddress+IMAGE_OPTIONAL_HEADER.SectionAlignment)>IMAGE_DATA_DIRECTORIES[0].VirtualAddress)):
                Export_table_adr=(IMAGE_DATA_DIRECTORIES[0].VirtualAddress-IMAGE_SECTION_HEADER.VirtualAddress) + \
                    IMAGE_SECTION_HEADER.PointerToRawData
                EXPORT_SECTION_HEADER=IMAGE_SECTION_HEADER


    if (IMAGE_DATA_DIRECTORIES[1].VirtualAddress!=0):
        SectionAdrEnd=Import_table_adr
        IMPORT_DIRECTORYs=[]
        IMPORT_THUNKs=[]
        while(1):
            SectionAdrStart=SectionAdrEnd
            SectionAdrEnd=SectionAdrStart+IMPORT_DIRECTORY_size
            IMPORT_DIRECTORY_values=struct.unpack(format_for_unpack(IMPORT_DIRECTORY_format),
                pe_file[SectionAdrStart:SectionAdrEnd])
            if(IMPORT_DIRECTORY_values==(0,)*5):
                break
            IMPORT_DIRECTORY=IMPORT_DIRECTORY_struct._make(IMPORT_DIRECTORY_values)
            print_import_directory(IMPORT_DIRECTORY,IMPORT_DIRECTORY_format,IMPORT_SECTION_HEADER,pe_file)
            IMPORT_DIRECTORYs.append(IMPORT_DIRECTORY)
            IMPORT_THUNK=[]

            thunk_adr=rva_to_offset(IMPORT_DIRECTORY.ImportNameTableRVA,IMPORT_SECTION_HEADER)
            print(("     IMPORT THUNK"))
            print(("     ============================================="))
            print(("      (  Address,     Data,        Value(Index,Name) )"))
            while(1):
                value=struct.unpack('<I',pe_file[thunk_adr:thunk_adr+4])[0]
                if(value==0):
                    break
                if((value & int("0x80000000",16))!=0 ):
                    IMPORT_THUNK.append(('0x{0:04x}'.format(thunk_adr),'ordinal',value))
                else:
                    hint_name_adr=rva_to_offset(value,IMPORT_SECTION_HEADER)
                    hint=struct.unpack('H',pe_file[hint_name_adr:hint_name_adr+2])[0]
                    name=read_string(pe_file,hint_name_adr+2)
                    IMPORT_THUNK.append(
                    ('0x{0:08x}'.format(thunk_adr),'0x{0:08x}'.format(value),'0x{0:04x}'.format(hint),name)
                    )
                print( "     ",IMPORT_THUNK[-1])
                thunk_adr+=4
            print()
            print()
            IMPORT_THUNKs.append(IMPORT_THUNK)

    if (IMAGE_DATA_DIRECTORIES[0].VirtualAddress!=0):
        SectionAdrEnd=Export_table_adr
        SectionAdrStart=SectionAdrEnd
        SectionAdrEnd=SectionAdrStart+IMAGE_EXPORT_DIRECTORY_size
        IMAGE_EXPORT_DIRECTORY_values=struct.unpack(format_for_unpack(IMAGE_EXPORT_DIRECTORY_format),
            pe_file[SectionAdrStart:SectionAdrEnd])
        IMAGE_EXPORT_DIRECTORY=IMAGE_EXPORT_DIRECTORY_struct._make(IMAGE_EXPORT_DIRECTORY_values)
        lib_name=read_string(pe_file,rva_to_offset(IMAGE_EXPORT_DIRECTORY.NameRVA,EXPORT_SECTION_HEADER))
        print_section(IMAGE_EXPORT_DIRECTORY,IMAGE_EXPORT_DIRECTORY_format,lib_name)

        print(("     Adress Table"))
        print(("     ==================================="))
        SectionAdrStart=rva_to_offset(IMAGE_EXPORT_DIRECTORY.AddressTableRVA,EXPORT_SECTION_HEADER)
        for i in range(IMAGE_EXPORT_DIRECTORY.NumberOfFunctions):
            fnRVA=struct.unpack(format_for_unpack([4]),
                pe_file[SectionAdrStart:SectionAdrStart+4])[0]
            SectionAdrStart += 4
            print( "     ",('0x{0:08x}'.format(fnRVA),"Function RVA"))

        print()
        print(("     Name Pointer Table RVA"))
        print(("     ==================================="))
        SectionAdrStart=rva_to_offset(IMAGE_EXPORT_DIRECTORY.NamePointerTableRVA,EXPORT_SECTION_HEADER)
        for i in range(IMAGE_EXPORT_DIRECTORY.NumberOfNames):
            nameRVA=struct.unpack(format_for_unpack([4]),
                pe_file[SectionAdrStart:SectionAdrStart+4])[0]
            SectionAdrStart += 4
            print( "     ",('0x{0:08x}'.format(nameRVA),read_string(pe_file,rva_to_offset(nameRVA,EXPORT_SECTION_HEADER))))

        print()
        print(("     Ordinal Table"))
        print(("     ==================================="))
        SectionAdrStart=rva_to_offset(IMAGE_EXPORT_DIRECTORY.OrdinalTableRVA,EXPORT_SECTION_HEADER)
        for i in range(IMAGE_EXPORT_DIRECTORY.NumberOfFunctions):
            ordinal=struct.unpack(format_for_unpack([2]),
                pe_file[SectionAdrStart:SectionAdrStart+2])[0]
            SectionAdrStart += 2
            print( "     ",('0x{0:04x}'.format(ordinal),"Function Ordinal"))
        print()









def print_section(sec_object,nv_format,descr=None):
    print( sec_object.__class__.__name__)
    print( "============================================")
    if descr:
        print( descr )
    for name,f in zip(sec_object._fields,nv_format):
        value=getattr(sec_object,name)
        print( name,":",)
        if(f<=4):
            print( ("0x%0."+str(f*2)+"X") % value)
        else:
            print( binascii.hexlify(value),value) #encode('utf-8')
    print()

def rva_to_offset(rva,SECTION_HEADER):
    return (rva-SECTION_HEADER.VirtualAddress) + \
        SECTION_HEADER.PointerToRawData

def read_string(bfile,address):
    s=''
    i=0
    while(1):

        character=struct.unpack('c',bfile[address+i:address+i+1])[0].decode('utf-8')
        if ord(character)==0:
            break
        s+=character
        i+=1
    return s
def print_import_directory(directory,nv_format,SECTION_HEADER,bfile):
        print( directory.__class__.__name__)
        print( "============================================")
        i=0
        print( read_string(bfile,rva_to_offset(directory.NameRVA,SECTION_HEADER)))
        for name,f in zip(directory._fields,nv_format):
            value=getattr(directory,name)
            print( name,":",)
            if(f<=4):
                print( ("0x%0."+str(f*2)+"X") % value,)
                if(i in [0,3,4]):
                    print( ("------>0x%0."+str(f*2)+"X") % rva_to_offset(value,SECTION_HEADER))

                else:
                    print()

            else:
                print( binascii.hexlify(value),value )#encode('utf-8')
            i=i+1
        print()

def format_for_unpack(format_list):
    f='<'
    for nbyte in format_list:
        if nbyte==1:
            f+='B'
        elif nbyte==2:
            f+='H'
        elif nbyte==4:
            f+='I'
        else:
            f+=str(nbyte)+'s'
    return f


if __name__ == '__main__':
    if len(sys.argv)==2:
        arg=sys.argv[1]
    else:
        arg=input()
    main(arg)
