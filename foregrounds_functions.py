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
from scipy import interpolate
from scipy import integrate as si
import sys

###############################################################################
###############################################################################

##### Functions that are used to simulate the radio foregrounds:

### Synchrotron: synchrotron
### Free-free: free_free
### AME: ame
### Thermal dust: thermal_dust
### Point sources: source_count_battye, temp_background_fixed_freq, cl_poisson_fixed_freq, map_poisson_fixed_freq_high_flow, cl_cluster_fixed_freq, point_sources
### CMB: cmb

###############################################################################
###############################################################################

##### Constants:

k_bol = 1.38064852e-23 # Boltzmann constant, in J K^-1
h_planck = 6.62607004e-34 # Planck constant, in m^2 kg s^-1
c = 299792458.   # speed of light, in m s^-1
jy_to_si = 1.e-26  # jansky to SI

###############################################################################
###############################################################################

##### Synchrotron

def synchrotron(frequencies, nside, model_template, spectral_index_model, curvature_index, curvature_reference_freq):

    """Galactic synchrotron cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: template and spectral index
    
    if model_template == 'haslam2014':
        synch_0 = hp.read_map('data/haslam408_dsds_Remazeilles2014_ns2048.fits')
        freq_0 = 0.408 # GHz
    else:
        sys.exit("Not a valid model!!!")

    if spectral_index_model == 'uniform':
        specind_0 = hp.read_map('data/synchrotron_specind_uniform.fits')
    elif spectral_index_model == 'mamd2008':
        specind_0 = hp.read_map('data/synchrotron_specind_mamd2008.fits')
    elif spectral_index_model == 'giardino2002':
        specind_0 = hp.read_map('data/synchrotron_specind_giardino2002.fits')
    else:
        sys.exit("Not a valid model!!!")
    
    ### Maps

    nchannels = frequencies.size
    
    synch = hp.ud_grade(synch_0, nside_out = nside)
    synch_0 = 0
    specind = hp.ud_grade(specind_0, nside_out = nside)
    specind_0 = 0

    maps = np.zeros((nchannels, hp.nside2npix(nside)))

    ### Frequency scaling
    
    for i in range(0, nchannels):
        maps[i, :] = 1000. * ((frequencies[i]) / freq_0)**(specind - 2.0 + curvature_index * np.log10(frequencies[i] / curvature_reference_freq)) * synch

    return maps # mK

###############################################################################
###############################################################################

##### Free-free

def free_free(frequencies, nside, model_template, temp_electron):

    """Galactic free-free cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: template
    
    if model_template == 'dickinson2003':
        halpha_0 = hp.read_map('data/onedeg_diff_halpha_JDprocessed_smallscales_2048.fits')
    else:
        sys.exit("Not a valid model!!!")

    ### Maps

    nchannels = frequencies.size
    
    halpha = hp.ud_grade(halpha_0, nside_out = nside, order_in = 'RING', order_out = 'RING')
    halpha_0 = 0

    maps = np.zeros((nchannels, hp.nside2npix(nside)))

    ### Factor a -- see Dickinson et al. 2003)

    factor_a = 0.366 * (frequencies)**0.1 * temp_electron**(-0.15) * (np.log(4.995 * 1.0e-2 * (frequencies)**(-1)) + 1.5 * np.log(temp_electron))

    ### Frequency scaling -- see Dickinson et al. 2003

    for i in range(0, nchannels):
        maps[i, :] = 8.396 * factor_a[i] * (frequencies[i])**(-2.1) * (temp_electron * 1.0e-4)**0.667 * 10**(0.029 / (temp_electron * 1.0e-4)) * 1.08 * halpha

    return maps # mK

###############################################################################
###############################################################################

##### AME

def ame(frequencies, nside, model_template, ame_ratio, ame_freq_in):

    """Galactic AME cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: template
    
    if model_template == 'planck_t353':
        tau_0 = hp.read_map('data/Planck_map_t353_small_scales.fits')
    else:
        sys.exit("Not a valid model!!!")

    ### Maps

    nchannels = frequencies.size
    
    tau = hp.ud_grade(tau_0, nside_out = nside)
    tau_0 = 0

    maps = np.zeros((nchannels, hp.nside2npix(nside)))   

    ### Frequency scaling: SPDUST -- see  Ali-Haimoud et al. 2009 and Silsbee et al. 2011

    spdust_file = 'data/spdust2_cnm.dat' # SPDUST code output file to obtain the frequency spectrum shape
    freq, flux = np.loadtxt(spdust_file, comments = ';', usecols = (0, 1), unpack = True)
    
    intp = interpolate.interp1d(freq, flux) # get the interpolation function of freq and flux from the spdust file 
    flux_in = intp(ame_freq_in)

    for i in range (0, nchannels):
        flux_out = intp(frequencies[i])
        scale = flux_out / flux_in * (ame_freq_in / frequencies[i])**2
        maps[i, :] = 1.e-3 * tau * ame_ratio * scale

    return maps # mK

###############################################################################
###############################################################################

##### Thermal Dust

def thermal_dust(frequencies, nside, model_template, spectral_index_model, temp_model):

    """Galactic thermal dust cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: template, spectral index, and temperature
    
    if model_template == 'gnilc_353':
        td_0 = hp.read_map('data/COM_CompMap_Dust-GNILC-F353_2048_R2.00_small_scales.fits')
        td_0 = 261.20305067644796 * td_0 # from 'MJy/sr' to 'muK_RJ'
        freq_in = 353.0  # GHz
    else:
        sys.exit("Not a valid model!!!")

    if spectral_index_model == 'gnilc_353':
        specind_0 = hp.read_map('data/COM_CompMap_Dust-GNILC-Model-Spectral-Index_2048_R2.00.fits')
    else:
        sys.exit("Not a valid model!!!")

    if temp_model == 'gnilc_353':
        temp_0 = hp.read_map('data/COM_CompMap_Dust-GNILC-Model-Temperature_2048_R2.00.fits') # K
    else:
        sys.exit("Not a valid model!!!")

    ### Maps

    nchannels = frequencies.size

    td = hp.ud_grade(td_0, nside_out = nside)
    td_0 = 0
    specind = hp.ud_grade(specind_0, nside_out = nside)
    specind_0 = 0
    temp = hp.ud_grade(temp_0, nside_out = nside)
    temp_0 = 0

    maps = np.zeros((nchannels, hp.nside2npix(nside)))

    ### Frequency scaling: modified blackbody -- see Planck Collaboration XI et al. 2013

    gamma = h_planck / (k_bol * temp)

    for i in range(0, nchannels):
        maps[i, :] = 1.e-3 * td * (frequencies[i] / freq_in)**(specind + 1) * ((np.exp(gamma * (freq_in * 1.e9)) - 1.) / (np.exp(gamma * (frequencies[i] * 1.e9)) - 1.))
    
    return maps  # mK

###############################################################################
###############################################################################

##### Point Sources

### Source count models

def source_count_battye(flux):

    """Source count for point sources at 1.4 GHz - see Battye et al 2013.
    """

    # Polynomial constants

    S_0 = 1.0
    N_0 = 1.0

    a_0 = 2.593
    a_1 = 9.333e-2
    a_2 = -4.839e-4
    a_3 = 2.488e-1
    a_4 = 8.995e-2
    a_5 = 8.506e-3

    # Nomalized flux

    ss = flux / S_0

    # Differential source count

    if flux <= 1.0e0: 
        dn_ds = N_0 * flux**(-2.5) * 10**(a_0 + a_1 * np.log10(ss) + a_2 * (np.log10(ss))**2 + a_3 * (np.log10(ss))**3 + a_4 * (np.log10(ss))**4 + a_5 * (np.log10(ss))**5)

    # Interpolate for high fluxes (>1.0e0 Jy)

    if flux > 1.0e0:
        par = np.array([2.59300238, -2.40632446])
        dn_ds = 10.**(par[0] + par[1] * np.log10(flux))

    # Output

    return dn_ds

### Background temperature

def temp_background_fixed_freq(flux_max, model_source_count):

    """Background temperature (in mK) at a fixed frequency (model dependent).
    """

    if flux_max <= 1.0e-6:
        sys.exit("Maximum flux has to be larger or equal to 1.0e-6 Jy.")

    if model_source_count == "battye2013":
        fixed_freq = 1.4e9   # Hz
        flux_min = 1.0e-6 # Minimum flux (in Jy)
        conversion_factor = 2 * k_bol * ((fixed_freq) / c)**2   # flux to temperature

	# To ensure the correct convergence of the integral, we divide it in decades (that is
	# why we have several integrations in what follows).

        if flux_max  <= 1.0e-3:
            N = 100000    
            s_integrand = np.linspace(flux_min, flux_max, N)
            s_function = np.zeros(N)
            for i in range(0, int(N)):
                s_function[i] = s_integrand[i] * source_count_battye(s_integrand[i])
            temp_ps = (1.0 / conversion_factor) * jy_to_si * (si.simps(s_function, s_integrand))
        elif 1.0e-3 < flux_max <= 1.0e1:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N)
            s_integrand_2 = np.linspace(1.0e-3, flux_max, N)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i] * source_count_battye(s_integrand_1[i])
                s_function_2[i] = s_integrand_2[i] * source_count_battye(s_integrand_2[i])
            temp_ps = (1.0 / conversion_factor)  * jy_to_si * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2))
        elif 1.0e1 < flux_max <= 1.0e3:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N, endpoint=True)
            s_integrand_2 = np.linspace(1.0e-3, 1.0e1, N, endpoint=True)
            s_integrand_3 = np.linspace(1.0e1, flux_max, N, endpoint=True)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            s_function_3 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i] * source_count_battye(s_integrand_1[i])
                s_function_2[i]= s_integrand_2[i] * source_count_battye(s_integrand_2[i])
                s_function_3[i]= s_integrand_3[i] * source_count_battye(s_integrand_3[i])
            temp_ps = (1.0 / conversion_factor) * jy_to_si * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2) + si.simps(s_function_3, s_integrand_3))
        elif flux_max > 1.0e3:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N, endpoint=True)
            s_integrand_2 = np.linspace(1.0e-3, 1.0e1, N, endpoint=True)
            s_integrand_3 = np.linspace(1.0e1, 1.0e3, N, endpoint=True)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            s_function_3 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i] * source_count_battye(s_integrand_1[i])
                s_function_2[i] = s_integrand_2[i] * source_count_battye(s_integrand_2[i])
                s_function_3[i] = s_integrand_3[i] * source_count_battye(s_integrand_3[i])
            temp_ps = (1.0 / conversion_factor) * jy_to_si * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2) + si.simps(s_function_3, s_integrand_3))

    return 1000.0 * temp_ps

### Poisson at low flux

def cl_poisson_fixed_freq(flux_max, model_source_count):

    """Poisson distribution of the sources (in mK^2) at a fixed frequency (model dependent).
    """

    if flux_max <= 1.0e-6:
        sys.exit("Maximum flux has to be larger or equal to 1.0e-6 Jy.")

    if model_source_count == "battye2013":
        fixed_freq = 1.4e9   # Hz
        flux_min = 1.0e-6 # Minimum flux (in Jy)
        conversion_factor = 2 * k_bol * ((fixed_freq)/ c)**2   # flux to temperature

	# To ensure the correct convergence of the integral, we divide it in decades (that is
	# why we have several integrations in what follows).

        if flux_max  <= 1.0e-3:
            N = 100000    
            s_integrand = np.linspace(flux_min, flux_max, N)
            s_function = np.zeros(N)
            for i in range(0, int(N)):
                s_function[i] = s_integrand[i]**2 * source_count_battye(s_integrand[i])
            aps_poisson = (1.0 / conversion_factor)**2 * jy_to_si**2 * (si.simps(s_function, s_integrand))
        elif 1.0e-3 < flux_max <= 1.0e1:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N)
            s_integrand_2 = np.linspace(1.0e-3, flux_max, N)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i]**2 * source_count_battye(s_integrand_1[i])
                s_function_2[i] = s_integrand_2[i]**2 * source_count_battye(s_integrand_2[i])
            aps_poisson = (1.0 / conversion_factor)**2 * jy_to_si**2 * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2))
        elif 1.0e1 < flux_max <= 1.0e3:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N, endpoint=True)
            s_integrand_2 = np.linspace(1.0e-3, 1.0e1, N, endpoint=True)
            s_integrand_3 = np.linspace(1.0e1, flux_max, N, endpoint=True)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            s_function_3 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i]**2 * source_count_battye(s_integrand_1[i])
                s_function_2[i]= s_integrand_2[i]**2 * source_count_battye(s_integrand_2[i])
                s_function_3[i]= s_integrand_3[i]**2 * source_count_battye(s_integrand_3[i])
            aps_poisson = (1.0 / conversion_factor)**2 * jy_to_si**2 * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2) + si.simps(s_function_3, s_integrand_3))
        elif flux_max > 1.0e3:
            N = 100000
            s_integrand_1 = np.linspace(flux_min, 1.0e-3, N, endpoint=True)
            s_integrand_2 = np.linspace(1.0e-3, 1.0e1, N, endpoint=True)
            s_integrand_3 = np.linspace(1.0e1, 1.0e3, N, endpoint=True)
            s_function_1 = np.zeros(N)
            s_function_2 = np.zeros(N)
            s_function_3 = np.zeros(N)
            for i in range(0, int(N)):
                s_function_1[i] = s_integrand_1[i]**2 * source_count_battye(s_integrand_1[i])
                s_function_2[i] = s_integrand_2[i]**2 * source_count_battye(s_integrand_2[i])
                s_function_3[i] = s_integrand_3[i]**2 * source_count_battye(s_integrand_3[i])
            aps_poisson = (1.0 / conversion_factor)**2 * jy_to_si**2 * (si.simps(s_function_1, s_integrand_1) + si.simps(s_function_2, s_integrand_2) + si.simps(s_function_3, s_integrand_3))
            
    return 1000.0**2 * aps_poisson

### Poisson at high flux (flux_min has to be the same as flux_max of poisson_fixed_freq) - (flux_max ~ 10^3 Jy)

def map_poisson_fixed_freq_high_flow(flux_min_hf, flux_max_hf, nside, model_source_count):

    """Poisson map for high fluxes (in mK) at a fixed frequency (model dependent).
    """

    omega_pix = (4.0 * np.pi) / (12.0 * nside**2)

    decades = np.floor(np.log10(flux_max_hf)) - np.floor(np.log10(flux_min_hf))
    dn_ds = np.zeros(10)
    map = np.zeros(hp.nside2npix(nside))

    if model_source_count == "battye2013":
        fixed_freq = 1.4e9   # Hz
        flux_min = 1.0e-6 # Minimum flux (in Jy)
        conversion_factor = 2 * k_bol * ((fixed_freq)/ c)**2   # flux to temperature 

	# We randomly distribute in the sky N of sources with flux S such that these
	# sources respect the underlying source count. This is done decade by decade.

        for i in range(0, int(decades)):
            s = np.linspace(flux_min_hf * 10**i, flux_min_hf * 10**(i + 1), 10)
            for j in range(0, 9):
                s_delta = np.linspace(s[j], s[j + 1], 10, endpoint=False)   
                for k in range(0, 10):
                    dn_ds[k] = source_count_battye(s_delta[k])
                delta_n = np.ceil(si.simps(dn_ds, s_delta))
                for l in range(0, int(delta_n)):
                    temp = (1.0 / conversion_factor) * jy_to_si * (1.0 / omega_pix) * np.random.uniform(low=s[j], high=s[j + 1])
                    map_index = np.random.randint(low=0, high=(hp.nside2npix(nside) - 1))
                    map[map_index] = temp

    return 1000.0 * map

### Clustering distribution

def cl_cluster_fixed_freq(ell, t_ps, model_source_count):

    """Clustering distribution of the sources (in mK^2) at a fixed frequency (model dependet).
    """

    if ell == 0:
        ell = 1.0e-3

    if model_source_count == "battye2013":        
        w_l = 1.8e-4 * ell**(-1.2)
        
    cl_cluster = w_l * t_ps**2
    
    return cl_cluster

def point_sources(frequencies, nside, model_source_count, max_flux_poisson_cl, max_flux_point_sources, spectral_index, spectral_index_std, add_clustering):

    """Extragalactic point sources cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: source count
   
    if model_source_count == "battye2013":
        fixed_freq = 1.4   # in GHz
    else:
        sys.exit("Not a valid model!!!")
    
    lmax = 3 * nside - 1

    ### Maps

    nchannels = frequencies.size
    maps = np.zeros((nchannels, hp.pixelfunc.nside2npix(nside)))

    ### Map generator (cl to map or dn / ds to map) and frequency scaling

    if add_clustering == True:

        cl_poisson = np.zeros(lmax + 1)
        cl_cluster = np.zeros(lmax + 1)
        t_ps = temp_background_fixed_freq(max_flux_point_sources, model_source_count)
        cl = cl_poisson_fixed_freq(max_flux_poisson_cl, model_source_count) # the same value for all ells                                                   
        poisson_high = map_poisson_fixed_freq_high_flow(max_flux_poisson_cl, max_flux_point_sources, nside, model_source_count) # template for the high flux sources 

        for l in range(0, lmax + 1):
            cl_poisson[l] = cl
            cl_cluster[l] = cl_cluster_fixed_freq(l, t_ps, model_source_count)

        map_poisson_0 = hp.synfast(cl_poisson, nside, lmax=lmax)
        map_cluster_0 = hp.synfast(cl_cluster, nside, lmax=lmax)

        alpha_array = np.zeros(hp.nside2npix(nside))

        for p in range(0, hp.nside2npix(nside)):
            alpha_array[p] = spectral_index_std * np.random.randn() + spectral_index

        for i in range(0, nchannels):                
            maps[i, :] = (frequencies[i] / fixed_freq)**(alpha_array) * (map_poisson_0 + map_cluster_0) +  (frequencies[i] / fixed_freq)**(alpha_array) * poisson_high + (frequencies[i] / fixed_freq)**(alpha_array) * t_ps
                        
    if add_clustering == False:

        cl_poisson = np.zeros(lmax + 1)
        t_ps = temp_background_fixed_freq(max_flux_point_sources, model_source_count)
        cl = cl_poisson_fixed_freq(max_flux_poisson_cl, model_source_count)    # the same value for all ells  
        poisson_high = map_poisson_fixed_freq_high_flow(max_flux_poisson_cl, max_flux_point_sources, nside, model_source_count) # template for the high flux sources

        for l in range(0, lmax + 1):
            cl_poisson[l] = cl
    
        map_poisson_0 = hp.synfast(cl_poisson, nside, lmax=lmax)

        alpha_array = np.zeros(hp.nside2npix(nside))

        for p in range(0, hp.nside2npix(nside)):
            alpha_array[p] = spectral_index_std * np.random.randn() + spectral_index_std

        for i in range(0, nchannels):
            maps[i, :] = (frequencies[i] / fixed_freq)**(alpha_array) * map_poisson_0 + (frequencies[i] / fixed_freq)**(alpha_array) * poisson_high + (frequencies[i] / fixed_freq)**(alpha_array) * t_ps
                    
    return maps # mK

###############################################################################
###############################################################################

##### CMB

def cmb(frequencies, nside, cmb_model):

    """CMB cube: it uses HEALPIX pixelization and is in mK.
    """

    ### Models: angular power spectrum
    
    if cmb_model == "standard":
        cl_cmb_0 = np.loadtxt('data/cmb_tt.txt')
        ell_0 = np.loadtxt('data/cmb_ell.txt')
        nside_0 =  2048
        lmax_0 = 3 * nside_0 - 1
    else:
        sys.exit("Not a valid model!!!")

    ### Maps

    nchannels = frequencies.size
    maps = np.zeros((nchannels, hp.pixelfunc.nside2npix(nside)))
    
    cl_cmb = np.zeros(lmax_0 + 1)

    for l in range(0, lmax_0 - 1):
        cl_cmb[l + 2] = 2. * np.pi * cl_cmb_0[l]  / (ell_0[l] * (ell_0[l] + 1.))

    map_cmb_ther_0 = 1.e-6 * hp.synfast(cl_cmb, nside_0)

    map_cmb_ther = hp.ud_grade(map_cmb_ther_0, nside_out = nside)
    map_cmb_ther_0 = 0 

    ### Frequency scaling -- from thermodynamic temperature to brightness temperature

    gamma = h_planck / k_bol # Planck / Boltzmann
    cmb_mean = 2.73 # K

    for i in range(0, nchannels):       
        maps[i, :] = ((gamma * (frequencies[i] * 1.e9) / cmb_mean)**2 * np.exp(gamma * (frequencies[i] * 1.e9) / cmb_mean) / (np.exp(gamma * (frequencies[i] * 1.e9) / cmb_mean) - 1.)**2) * (map_cmb_ther[:]) # relationship between brightness temperature and thermodynamic temperature
        
    return 1.e3 * maps # mK
