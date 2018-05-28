###############################################################################
###############################################################################
#
# This piece of code has been developed by
# 	Lucas C. Olivari
# and is part of the IM Foreground Sky Model (IMFSM).
#
# For more information about the IMFSM contact 
# 	Lucas C. Olivari (lolivari@if.usp.br)
#
# May, 2018
#
###############################################################################
###############################################################################

import ConfigParser

def ConfigSectionMap(Config, initial_file, section):
    dict1 = {}
    Config.read(initial_file)
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1

def ConfigGetBoolean(Config, initial_file, section, entry):

    Config.read(initial_file)

    return Config.getboolean(section, entry)
