from barak.io import parse_config, readtxt
from scipy.integrate import simps
import numpy as np
from barak.constants import Ryd_Ang, pi, hplanck

import os

def get_data_path():
    """ Return the path to the data directory for this package.
    """
    return os.path.abspath(__file__).rsplit('/', 1)[0] + '/data/'

def lya_gamma_faucher(z):
    """ The photoionisation rate of HI / 10^-12 

    Units are photons/s

    from Faucher-Giguere et al. 2009 Table 2,
    http://adsabs.harvard.edu/abs/2009ApJ...703.1416F
    """
    gamma = (0.0384, 0.0728, 0.1295, 0.2082, 0.3048, 0.4074, 0.4975,
             0.5630, 0.6013, 0.6142, 0.6053, 0.5823, 0.5503, 0.5168,
             0.4849, 0.4560, 0.4320, 0.4105, 0.3917, 0.3743, 0.3555,
             0.3362, 0.3169, 0.3001, 0.2824, 0.2633, 0.2447, 0.2271,
             0.2099)

    zvals = 0.25 * np.arange(len(gamma))

    return np.interp(z, zvals, gamma)


def hcross_section(energy):
    """ The photoionization cross section of HI in units of cm^2
    at the given energies.

    Energy must have units of Rydbergs. From Ferland & Osterbrock page
    20. """
    energy = np.atleast_1d(energy)
    a_nu = np.zeros(len(energy), float)

    A0 = 6.30e-18

    c0 = energy > 1
    if c0.any():
        eps = np.sqrt(energy[c0] - 1.)
        num = np.exp(4. - ((4 * np.arctan(eps)) / eps))
        den = 1. - np.exp(-2. * pi/eps)
        a_nu[c0] = A0 * (1. / energy[c0])**4 * num / den

    a_nu[energy == 1] = A0

    return a_nu

def find_gamma(energy, jnu):
    """ Find the photoionization rate / 1e12 given the spectrum as a
    function of energy.

    in units of photons / s.

    This is copying JFH's code in cldy_cuba_jfh.pro.
    """

    sigma_nu = hcross_section(energy)
    # output is in units of 1e12
    integrand = 4. * pi * jnu * sigma_nu / hplanck * 1e12

    log_energy = np.log10(energy)
    isort = np.argsort(log_energy)
    # don't understand why this works...
    gamma = np.log(10) * simps(integrand[isort], x=log_energy[isort])
    return gamma


def read_observed(filename):
    """ Read a config-style file with ions and column densities.

    Each ion has three lines, first line is log10 of N (cm^-2),second
    and third lines are lower and upper errors.
    """
    obs = parse_config(filename)
    for k in obs:
        vals = map(float, obs[k].split())
        if len(vals) == 2:
            obs[k] = vals[0], vals[1], vals[1]
        elif len(vals) == 3:
            obs[k] = tuple(vals)
        else:
            raise ValueError('Error parsing entry %s' % obs[k])

    return obs

def get_ratio(a, b):
    """ Measure minimum and maximum ratio for a/b"""
    return (a[0]-a[1]) - (b[0]+b[2]), a[0]-b[0], (a[0]+a[2]) - (b[0]-b[1])


def calc_uvb(redshift, cuba_name, match_fg=False):

    """ Calculate the UV background for a Haardt-Madau model.

    Returns a dictionary with information about the UV model:

    =======  ====================================================
    energy   energy in Rydbergs 
    Jnu      log10 of the Intensity at each energy (erg/s/cm^2/Hz/ster)
    Jnu_912  Mean intensity at 1 Rydberg (912 Ang or 13.6 eV) in
             erg/s/cm^2/Hz/ster.
    =======  ====================================================
    """

    if cuba_name.endswith('UVB.out'):
        redshifts, wave, jnu_z = read_cuba(cuba_name)
    else:
        redshifts, wave, jnu_z = read_XIDL_cuba(cuba_name)
    jnu0 = interp_cuba_to_z(redshift, redshifts, jnu_z)
    energy0 = Ryd_Ang / wave  # ergs
    isort = energy0.argsort()
    energy = energy0[isort]
    jnu = jnu0[isort]

    # scale to match faucher-giguere background
    if match_fg:
        gamma_fg = lya_gamma_faucher(redshift)
        mult = gamma_fg / find_gamma(energy, jnu)
        print 'Scaling Jnu by %.3g to match FG gamma' % mult
        jnu *= mult

    logjnu = np.where(jnu > 1e-30, np.log10(jnu), -30)


    return dict(energy=energy, logjnu=logjnu)

def calc_local_jnu(wa, logFtot, distkpc, f_esc=1, NHI_fesc=1e20):
    """
    Parameters
    ----------
    wa : array_like, shape(N,)
      Wavelengths for the input spectrum in Angstroms.
    logFtot : array_like, shape (N,)
      log10 of the total flux from the galaxy in erg/s/Ang.
    distkpc : float
      Distance at which to calculate Jnu in kpc.
    f_esc : float
      Escape fraction (<= 1). Default 1.
    NHI : float
      See Cantalupo 2010, MNRAS, 403, L16. Default 1e20.

    returns
    -------
    nu, logJnu : arrays of shape (N,)
      The frequency (Hz) and log10 of the intensity of the output spectrum
      (erg/s/cm^2/Hz/ster) at the given distance.
    """
    
    from barak.constants import kpc, c, hplanck, Ryd

    wa_cm = wa * 1e-8

    # erg/s/cm
    fwatot_cm = 10**logFtot * 1e8
     
    # total area in cm^2 over which this energy is spread
    area = 4 * np.pi * (distkpc * kpc)**2
     
    fwa_local = fwatot_cm / area
     
    # fnu = lambda^2 / c * f_lambda
    Fnu = wa_cm**2 / c * fwa_local

    # erg/s/cm^2/Hz/ster
    Jnu = Fnu / (4 * np.pi)

    nu = c / wa_cm
    nu = nu[::-1]
    Jnu = Jnu[::-1]
    if f_esc < 1:
        sigma_nu = hcross_section(nu * hplanck / Ryd) 
        cond = (nu * hplanck / Ryd) > 1.
        Jnu[cond] = Jnu[cond] * (
            f_esc + (1 - f_esc) * np.exp(-1 * sigma_nu[cond] * NHI_fesc) )

    return nu, np.log10(Jnu)


def read_starburst99(filename):
    """ Read a spectrum output by starburst99.

    Takes the spectrum with the latest time.

    Returns
    -------
    wa, F: ndarrays, shape (N,)
      wavelength in Angstroms and log10(Flux in erg/s/A)
    """
    T = np.genfromtxt(filename, skiprows=6, names='time,wa,tot,star,neb')
    
    times = np.unique(T['time'])
    
    cond = T['time'] == times[-1]
    wa = T['wa'][cond]
    F = T['tot'][cond]              
    return wa, F
    

def read_XIDL_cuba(filename):
    """ Parse a Haart & Madau CUBA file as given in XIDL.

    return jnu as a function of wavelength and redshift. Removes
    duplicate wavelength points.

    Returns
    -------
    redshifts : array of floats with shape (N,)
    wa : array of floats with shape (M,)
      Wavelengths in Angstroms.
    jnu : array of floats with shape (N, M)
      Flux in erg/s/Hz/cm^2/sr.
    """
    
    fh = open(filename)
    rows = fh.readlines()
    fh.close()
     
    redshifts, wa, jnu = [], [], []

    # this is the row index
    i = 0
    # another index keep track of which set of redshifts we're at
    indz = 0
    while i < len(rows):
        # Read the line of redshifts
        zvals = [float(val) for val in rows[i].split()[2:]]
        redshifts.extend(zvals)
        # need a jnu array for each redshift
        jnu.extend([] for z in zvals)
        i += 1
        # Read jnu values and wavelengths until we hit another row of
        # redshifts or the end of the file
        while i < len(rows) and not rows[i].lstrip().startswith('#'):
            #print i, rows[i]
            items = []
            for val in rows[i].split():
                try:
                    items.append(float(val))
                except ValueError:
                    if not(val.endswith('-310') or
                           val.endswith('-320')):
                        print 'replacing jnu value of ', val, 'with 0'
                    items.append(0.0)
            if indz == 0:
                wa.append(items[0])
            for j,jnu_val in enumerate(items[1:]):
                jnu[indz+j].append(jnu_val)
            i += 1
     
        indz += len(zvals)

    redshifts, wa, jnu = (np.array(a) for a in (redshifts,wa,jnu))

    # remove duplicates
    _, iuniq = np.unique(wa, return_index=True)
    wa1 = wa[iuniq]
    jnu1 = jnu[:, iuniq]

    return redshifts, wa1, jnu1


def read_cuba(filename):
    """ Parse a Haart & Madau CUBA file.

    return jnu as a function of wavelength and redshift. Removes
    duplicate wavelength points.

    Returns
    -------
    redshifts : array of floats with shape (N,)
    wa : array of floats with shape (M,)
      Wavelengths in Angstroms.
    jnu : array of floats with shape (N, M)
      Flux in erg/s/Hz/cm^2/sr.
    """

    # first read the redshifts
    fh = open(filename)
    for row in fh:
        r = row.strip()
        if not r or r.startswith('#'):
            continue
        redshifts = np.array(map(float, r.split()))
        break

    # then the wavelenths and fnu values
    cols = readtxt(filename, skip=1)
    assert len(cols) == len(redshifts) + 1
    wa = cols[0]
    jnu = np.array(cols[1:])

    # remove duplicates
    _, iuniq = np.unique(wa, return_index=True)
    wa1 = wa[iuniq]
    jnu1 = jnu[:, iuniq]

    return redshifts, wa1, jnu1


def interp_cuba_to_z(ztarget, redshifts, jnu):
    """ Find jnu as a function of wavelength at the target redshift.

    Linearly interpolates between redshifts for the 2d array jnu.

    Inputs
    ------
    ztarget : float
    redshifts : array of floats, shape (N,)
    jnu : array of floats, shape (N, M)
    
    Returns
    -------
    jnu_at_ztarget : array of floats shape (M,)
    """
    Nwa = jnu.shape[1]
    jnu_out = [np.interp(ztarget, redshifts, jnu.T[i]) for i in xrange(Nwa)]
    return np.array(jnu_out)

