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

import numpy as np
import healpy as hp
import ConfigParser
from astropy.io import fits as pyfits
import os

import foregrounds_functions
import misc_functions

##### Read parameters.ini

Config = ConfigParser.ConfigParser()
initial_file = "parameters.ini"

##### Generate each of the asked foreground cubes plus the total foreground cube

### Experimental setup

freq_min = misc_functions.ConfigSectionMap(Config, initial_file, "General")['freq_min']
freq_width = misc_functions.ConfigSectionMap(Config, initial_file, "General")['freq_width']
nchannels = misc_functions.ConfigSectionMap(Config, initial_file, "General")['nchannels']
nside = misc_functions.ConfigSectionMap(Config, initial_file, "General")['nside']
output_suffix = misc_functions.ConfigSectionMap(Config, initial_file, "General")['output_suffix']

freq_min = float(freq_min)
freq_width = float(freq_width)
nchannels = int(float(nchannels))
nside = int(float(nside))

frequencies = freq_min + np.arange(nchannels) * freq_width

### Create output directory

path_in = os.path.realpath(__file__)
directory_in = os.path.dirname(path_in)
directory_out = directory_in + '/output'

if not os.path.exists(directory_out):
    os.makedirs(directory_out)

### Emissions

total = np.zeros((nchannels, hp.nside2npix(nside)))

# Synchrotron

if misc_functions.ConfigGetBoolean(Config, initial_file, "Synchrotron", 'simulate') == True:
    
    model_template = misc_functions.ConfigSectionMap(Config, initial_file, "Synchrotron")['model_template']
    spectral_index_model = misc_functions.ConfigSectionMap(Config, initial_file, "Synchrotron")['spectral_index_model']
    curvature_index = misc_functions.ConfigSectionMap(Config, initial_file, "Synchrotron")['curvature_index']
    curvature_reference_freq = misc_functions.ConfigSectionMap(Config, initial_file, "Synchrotron")['curvature_reference_freq']

    curvature_index = float(curvature_index)
    curvature_reference_freq = float(curvature_reference_freq)
    
    cube = foregrounds_functions.synchrotron(frequencies, nside, model_template, spectral_index_model, curvature_index, curvature_reference_freq)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "Synchrotron", 'save_cube') == True:

        pyfits.writeto('output/synch_cube_' + output_suffix + '.fits', cube, overwrite = True)

    cube = 0

# Free-free
    
if misc_functions.ConfigGetBoolean(Config, initial_file, "FreeFree", 'simulate') == True:

    model_template = misc_functions.ConfigSectionMap(Config, initial_file, "FreeFree")['model_template']
    temp_electron = misc_functions.ConfigSectionMap(Config, initial_file, "FreeFree")['temp_electron']

    temp_electron = float(temp_electron)

    cube = foregrounds_functions.free_free(frequencies, nside, model_template, temp_electron)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "FreeFree", 'save_cube') == True:

        pyfits.writeto('output/free_free_cube_' + output_suffix + '.fits', cube, overwrite = True)

    cube = 0

# AME

if misc_functions.ConfigGetBoolean(Config, initial_file, "AME", 'simulate') == True:

    model_template = misc_functions.ConfigSectionMap(Config, initial_file, "AME")['model_template']
    ame_ratio =  misc_functions.ConfigSectionMap(Config, initial_file, "AME")['ame_ratio']
    ame_freq_in =  misc_functions.ConfigSectionMap(Config, initial_file, "AME")['ame_freq_in']

    ame_ratio = float(ame_ratio)
    ame_freq_in = float(ame_freq_in)

    cube = foregrounds_functions.ame(frequencies, nside, model_template, ame_ratio, ame_freq_in)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "AME", 'save_cube') == True:

        pyfits.writeto('output/ame_cube_' + output_suffix + '.fits', cube, overwrite = True)

    cube = 0

# Thermal dust

if misc_functions.ConfigGetBoolean(Config, initial_file, "ThermalDust", 'simulate') == True:

    model_template = misc_functions.ConfigSectionMap(Config, initial_file, "ThermalDust")['model_template']
    spectral_index_model = misc_functions.ConfigSectionMap(Config, initial_file, "ThermalDust")['spectral_index_model']
    temp_model = misc_functions.ConfigSectionMap(Config, initial_file, "ThermalDust")['temp_model']

    cube = foregrounds_functions.thermal_dust(frequencies, nside, model_template, spectral_index_model, temp_model)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "ThermalDust", 'save_cube') == True:

        pyfits.writeto('output/thermal_dust_cube_' + output_suffix + '.fits', cube, overwrite = True)

    cube = 0

# Point Sources

if misc_functions.ConfigGetBoolean(Config, initial_file, "PointSources", 'simulate') == True:

    model_source_count = misc_functions.ConfigSectionMap(Config, initial_file, "PointSources")['model_source_count']
    max_flux_poisson_cl = misc_functions.ConfigSectionMap(Config, initial_file, "PointSources")['max_flux_poisson_cl']
    max_flux_point_sources = misc_functions.ConfigSectionMap(Config, initial_file, "PointSources")['max_flux_point_sources']
    spectral_index = misc_functions.ConfigSectionMap(Config, initial_file, "PointSources")['spectral_index']
    spectral_index_std = misc_functions.ConfigSectionMap(Config, initial_file, "PointSources")['spectral_index_std']

    add_clustering = misc_functions.ConfigGetBoolean(Config, initial_file, "PointSources", 'add_clustering')

    max_flux_poisson_cl = float(max_flux_poisson_cl)
    max_flux_point_sources = float(max_flux_point_sources)
    spectral_index = float(spectral_index)
    spectral_index_std = float(spectral_index_std)

    cube = foregrounds_functions.point_sources(frequencies, nside, model_source_count, max_flux_poisson_cl, max_flux_point_sources, spectral_index, spectral_index_std, add_clustering)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "PointSources", 'save_cube') == True:

        pyfits.writeto('output/point_sources_cube_' + output_suffix + '.fits', cube, overwrite = True)

    cube = 0

# CMB

if misc_functions.ConfigGetBoolean(Config, initial_file, "CMB", 'simulate') == True:

    cmb_model = misc_functions.ConfigSectionMap(Config, initial_file, "CMB")['cmb_model']

    cube = foregrounds_functions.cmb(frequencies, nside, cmb_model)

    total = total + cube

    if misc_functions.ConfigGetBoolean(Config, initial_file, "CMB", 'save_cube') == True:

        pyfits.writeto('output/cmb_cube_' + output_suffix + '.fits', cube, overwrite = True)
     
    cube = 0
pyfits.writeto('output/foreground_cube_' + output_suffix + '.fits', total, overwrite = True)
