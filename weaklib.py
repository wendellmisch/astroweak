########## library containing useful functions for the weak interaction in nuclei ##########
import math,numpy,os,re,scipy,sys
from scipy import optimize

##### begin list of functions #####
## atomic_mass                   Atomic mass of nuclide
## avg_weight                    Statistical weight for contributers to high energy average state
## BGT                           Compute BGT from log(ft) value
## BGT_FFN                       Compute BGT from log(ft) value using FFN values of avec_to_vec and B_to_ft
## choose                        Choose an element from a list
## deltaM                        Mass difference between two nuclei
## electron_chemical_potential   Chemical potential of electrons in specified environment
## electron_fermi_energy         Fermi energy of electrons in specified environment
## fortran_exp                   Format a number as 0.xEx
## ft                            Compute 1/ft from B
## ft_FFN                        Compute 1/ft from B using FFN values of avec_to_vec and B_to_ft
## get_energy                    Energy of specified state from lpt file
## get_lpt                       Choose an lpt file
## get_lptstates                 List of states from an lpt file
## get_lptexpstates              List of experimentally measured states from an lpt file
## get_n                         Number of the highest energy state less than specfied energy for specified J, T
## get_strength                  GT strength distributions for all computed states of specified nucleus
## get_strengthF                 F strength distributions for all computed states of specified nucleus
## histogram                     Weighted histogram
## imbalance                     Imbalance between inputs
## kernel_cc                     Kernel for charged current phase space integral
## logft                         log(ft) value from Q-value and half life
## nuc_rad                       Nuclear radius
## pair_erate                    Energy loss rate via nuclear de-excitation neutrino pairs
## partfcn                       Partition function computed from lpt
## partfcn_partial               Contribution to partition function from range of energy
## rates_cc                      Charged current reaction and energy loss rates
## sci_not                       Converts numbers to scientific notation
## spectrum_cc                   Charged current nuclear neutrino spectrum
## spectrum_nc                   Neutrino spectrum for unblocked de-excitation
## thermal_strength              Thermal strength distribution for specified nucleus
## user_input                    Gets user input and converts it to an appropriate format

##### end list of functions #####


##### begin channels #####
## A list of (possibly) supported reaction channels.
## Useful for keeping various programs on the same page.
channels = ['gt-','gt+','gt3']

##### end channels #####


##### begin constants #####
## Some handy constants and parameters.
alpha = 0.0072973525664             ## fine structure constant
avec_to_vec = (-1.2694)**2          ## (g_A/g_V)^2.  square of axial-vector-to-vector coupling ratio
avogadro = 6.0221409*10**23         ## Avogadro's number
B_to_ft = 6143.0                    ## B = B_F + (g_A/g_V)^2*B_GT = B_to_ft/ft.  K/g_V^2 in Cole et al (2012).  Value from Hardy and Towner (2009).
C = 2.99792458*10**23               ## speed of light (fm/s)
G_F = 1.1663787*10**(-11)           ## Fermi constant (MeV^-2)
hbar = 6.582119514*10**(-22)        ## reduced Planck's constant (MeV*s)
kB = 8.6173324*10**(-11)            ## Boltzmann constant (MeV/K)
me = 0.510998910                    ## electron mass (MeV)
MeV_per_erg = 624150.647996         ## conversion facter between MeV and ergs
m_sun = 1.9891*10**33               ## solar mass in grams
pi = math.pi                        ## 3.14159265...
quench = 0.764**2                   ## USDB Hamiltonian quenching factor; Richter, Mkhize, and Brown (2008)

## Dictionary of atomic masses.
atomic_masses = {
        (21,7): 25.230, (21,8): 8.062, (21,9): -0.0476, (21,10): -5.73172, (21,11): -2.1843, (21,12): 10.912,
        (22,6): 52.600, (22,7): 32.080, (22,8): 9.280, (22,9): 2.794, (22,10): -8.02435, (22,11): -5.1822, (22,12): -0.3968, (22,13): 18.180, (22,14): 32.160,
        (23,7): 37.700, (23,8): 14.620, (23,9): 3.330, (23,10): -5.1537, (23,11): -9.52950, (23,12): -5.4727, (23,13): 6.767, (23,14): 23.770,
        (24,8): 19.000, (24,9): 7.540, (24,10): -5.948, (24,11): -8.41762, (24,12): -13.93340, (24,13): -0.055, (24,14): 10.755,
        (25,9): 11.270, (25,10): -2.060, (25,11): -9.3575, (25,12): -13.19275, (25,13): -8.9158, (25,14): 3.825,
        (26,9): 18.290, (26,10): 0.430, (26,11): -6.902, (26,12): -16.21451, (26,13): -12.21032, (26,14): -7.145, (26,15): 10.970,
        (27,9): 25.100, (27,10): 7.090, (27,11): -5.580, (27,12): -14.58654, (27,13): -17.19686, (27,14): -12.38503, (27,15): -0.750, (27,16): 17.510,
        (28,10): 11.280, (28,11): -1.030, (28,12): -15.0188, (28,13): -16.85058, (28,14): -21.49283, (28,15): -7.161, (28,16): 4.070,
        (29,10): 18.000, (29,11): 2.620, (29,12): -10.660, (29,13): -18.2155, (29,14): -21.89506, (29,15): -16.9519, (29,16): -3.160,
        (30,10): 22.200, (30,11): 8.590, (30,12): -8.880, (30,13): -15.872, (30,14): -24.43292, (30,15): -20.2006, (30,16): -14.063,
        (31,11): 12.660, (31,12): -3.220, (31,13): -14.954, (31,14): -22.94899, (31,15): -24.4410, (31,16): -19.0449, (31,17): -7.060, (31,18): 11.300,
        (32,11): 18.300, (32,12): -0.800, (32,13): -11.060, (32,14): -24.0809, (32,15): -24.3053, (32,16): -26.01594, (32,17): -13.331, (32,18): -2.180,
        (33,11): 25.500, (33,12): 5.200, (33,13): -8.500, (33,14): -20.492, (33,15): -26.3377, (33,16): -26.58620, (33,17): -21.0035, (33,18): -9.380,
        (34,11): 32.500, (34,12): 8.500, (34,13): -2.860, (34,14): -19.957, (34,15): -24.558, (34,16): -29.93181, (34,17): -24.43961, (34,18): -18.378,
        (35,11): 41.200, (35,12): 16.300, (35,13): -0.060, (35,14): -14.360, (35,15): -24.8576, (35,16): -28.84633, (35,17): -29.01351, (35,18): -23.0482, (35,19): -11.167, (35,20): 4.440
        }

## Dictionary of atomic names.
atomic_names = {
        1:'H', 2:'He',
        3:'Li', 4:'Be', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 10:'Ne',
        11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 18:'Ar',
        19:'K', 20:'Ca', 21:'Sc', 22:'Ti', 23:'V', 24:'Cr', 25:'Mn', 26:'Fe', 27:'Co', 28:'Ni', 29:'Cu', 30:'Zn', 31:'Ga', 32:'Ge', 33:'As', 34:'Se', 35:'Br', 36:'Kr',
        37:'Rb', 38:'Sr', 39:'Y', 40:'Zr', 41:'Nb', 42:'Mo', 43:'Tc', 44:'Ru', 45:'Rh', 46:'Pd', 47:'Ag', 48:'Cd', 49:'In', 50:'Sn', 51:'Sb', 52:'Te', 53:'I', 54:'Xe',
        55:'Cs', 56:'Ba', 57:'La', 58:'Ce', 59:'Pr', 60:'Nd', 61:'Pm', 62:'Sm', 63:'Eu', 64:'Gd', 65:'Tb', 66:'Dy', 67:'Ho', 68:'Er', 69:'Tm', 70:'Yb', 71:'Lu'
        }
##### end constants #####


##### begin density-temperature grids #####
## FFN temps [T9 (log[k])]: 0.01 (7), 0.1 (8), 0.2 (8.3), 0.4 (8.6), 0.7 (8.85), 1.0 (9), 1.5 (9.18), 2.0 (9.3), 3.0 (9.48), 5.0 (9.7), 10.0 (10), 30.0 (10.48)
## grid[i] = [rhoYe i, array([temp1,temp2,...])]

FFNtemp = numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])    ## T9
FFNrhoYe = 10.**numpy.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.])    ## g/cm^3

## Abridged FFN grid.
abrFFNgrid = [
        [ 10**1., numpy.array([0.01, 0.1, 0.2, 0.4]) ],                                             ## 4
        [ 10**2., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0]) ],                                   ## 6
        [ 10**3., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]) ],                         ## 8
        [ 10**4., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]) ],                    ## 9
        [ 10**5., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]) ],               ## 10
        [ 10**6., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],   ## 12
        [ 10**7., numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],   ## 12
        [ 10**8., numpy.array([0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],         ## 11
        [ 10**9., numpy.array([0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],              ## 10
        [ 10**10.,numpy.array([2.0, 3.0, 5.0, 10.0, 30.0]) ],                                       ## 5
        [ 10**11.,numpy.array([5.0, 10.0, 30.0]) ]                                                  ## 3
        ]
        
## Complement FFN grid (FFN grid - abridged FFN grid).
compFFNgrid = [
        [ 10**1., numpy.array([0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],                        ## 8
        [ 10**2., numpy.array([1.5, 2.0, 3.0, 5.0, 10.0, 30.0]) ],                                  ## 6
        [ 10**3., numpy.array([3.0, 5.0, 10.0, 30.0]) ],                                            ## 4
        [ 10**4., numpy.array([5.0, 10.0, 30.0]) ],                                                 ## 3
        [ 10**5., numpy.array([10.0, 30.0]) ],                                                      ## 2
        [ 10**8., numpy.array([0.01]) ],                                                            ## 1
        [ 10**9., numpy.array([0.01, 0.1]) ],                                                       ## 2
        [ 10**10.,numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5]) ],                              ## 7
        [ 10**11.,numpy.array([0.01, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]) ]                     ## 9
        ]
        
## FFN grid.
FFNgrid = [
        [ 10**1., FFNtemp ],
        [ 10**2., FFNtemp ],
        [ 10**3., FFNtemp ],
        [ 10**4., FFNtemp ],
        [ 10**5., FFNtemp ],
        [ 10**6., FFNtemp ],
        [ 10**7., FFNtemp ],
        [ 10**8., FFNtemp ],
        [ 10**9., FFNtemp ],
        [ 10**10., FFNtemp ],
        [ 10**11., FFNtemp ]
        ]

##### end density-temperature grids #####


##### begin atomic_mass #####
## Returns atomic mass excess in MeV.
## A is nuclear mass number.  Can be any data type that will cast to int.
## Z is proton number.  Can be any data type that will cast to int.
def atomic_mass(A,Z):
    try:
        A = int(A)
    except:
        print '\"{}\" is invalid input for A in mass lookup.'.format(A)
        return None
        
    try:
        Z = int(Z)
    except:
        print '\"{}\" is invalid input for Z in mass lookup.'.format(Z)
        return None
        
    M = atomic_masses.get((A,Z))
    if M == None:
        print 'No mass found for A,Z={},{}.'.format(A,Z)
    
    return M
    
##### end atomic_masses #####


##### begin avg_weight #####
# Computes the average statistical weight for a range of states.
# First finds partial partition function for given range,
# then sums spin degeneracies of computed states in that range and divides by that sum.
# Computed states should then be multiplied by their degeneracy and the result from avg_weight().
# T (float) is temperature (MeV).  May be scalar or array.
# lpt_name (str) is file name for lpt.
# AZ (str) is 'atomic mass,atomic number'
# channel (str) is the reaction channel
# nuc_dir (str) is the directory containing the 'overlaps' directory
# E_min and E_max (float) set the energy range of accepted states.
# Tz (float) is z-projection of isospin.  May be neglected if lpt file contains no states with bad isospins
def avg_weight(T,lpt_name,AZ,channel,E_min=0,E_max=float('inf'),nuc_dir='./',Tz=None):
    if not lpt_name.startswith(nuc_dir):
        lpt_name = nuc_dir + lpt_name
    
    Z = int(AZ.split(',')[1])
        
    Gp = partfcn_partial(T,lpt_name,E_min=E_min,E_max=E_max,Tz=Tz)
    
    spinsum = 0
    for dao in os.listdir(nuc_dir+'overlaps/A,Z={}'.format(AZ)):
        if os.path.splitext(dao)[1] == '.dao':
            if dao.split()[0] == channel and dao.split()[1].endswith(str(Z)):
                J = float(dao.split()[3].split('=')[1])
                T = float(dao.split()[4].split('=')[1])
                n = dao.split()[5].split('=')[1].split('.')[0]
                E = get_energy(Tz,J,T,n,lpt_name=lpt_name)
                
                if E_min <= E < E_max:
                    spinsum = spinsum + 2*J+1

    weight = Gp/spinsum
    return weight

##### end avg_weight #####


##### begin BGT #####
# Takes log(ft) value and BF, returns BGT.  If Ji and Jf are specified, also returns reverse reaction BGT.
# logft (float) is the measured log(ft) value.
# Ji is the initial state spin.
# Jf is the final state spin.
# BF (float or int) is the Fermi transition strength.
def BGT(logft,Ji=None,Jf=None,BF=0.):
    ft = 10**(logft)
    
    # BGT for forward reaction
    BGTf = (B_to_ft/ft - BF)/avec_to_vec
    
    # FFN values of B_to_ft and avec_to_vec in following line.
    #BGTf = 10**(3.596)*(10**(-logft)-BF*10**(-3.791))
    
    # BGT for reverse reaction
    if Ji!=None and Jf!=None:
        BGTr = BGTf * (2.*Ji+1)/(2.*Jf+1)
    else:
        BGTr = None
    
    #print '\nft: {}\nJi: {}\nJf: {}\nBF: {}'.format(logft,Ji,Jf,BF)
    
    return (BGTf,BGTr)
##### end BGT #####


##### begin BGT_FFN #####
# Takes log(ft) value and BF, returns BGT using FFN values of weak constants.  If Ji and Jf are specified, also returns reverse reaction BGT.
# logft (float) is the measured log(ft) value.
# Ji is the initial state spin.
# Jf is the final state spin.
# BF (float or int) is the Fermi transition strength.
def BGT_FFN(logft,Ji=None,Jf=None,BF=0.):
    ft = 10**(logft)
    # BGT for forward reaction
    BGTf = 10**(3.596)*(10**(-logft)-BF*10**(-3.791))
    
    # BGT for reverse reaction
    if Ji!=None and Jf!=None:
        BGTr = BGTf * (2.*Ji+1)/(2.*Jf+1)
    else:
        BGTr = None
    
    print '\nft: {}\nJi: {}\nJf: {}\nBF: {}'.format(logft,Ji,Jf,BF)
    
    return (BGTf,BGTr)
##### end BGT #####


##### begin choose #####
# Prompts user to select one item from a list.
# Returns selection or None if no item is selected.
def choose(thing,options,mustchoose=True,nochoice='to make no selection'):
    print ''
    for o in range(len(options)):
        print str(o+1)+': '+str(options[o])
    if mustchoose:
        text = 'Choose '+thing+' by number: '
    else:
        text = 'Choose '+thing+' by number (<enter> '+nochoice+'): '
    while True:
        choice = raw_input(text)
        try:
            choice = options[int(choice) - 1]
            break
        except:
            if not choice and not mustchoose:
                choice = None
                break
            else:
                print 'Invalid choice.'
    return choice
##### end choose #####


##### begin deltaM #####
## Computes nuclear mass difference M1 - M2 from experimental data.
## A1 is A of nucleus 1.
## Z1 is Z of nucleus 1.
## A2 is A of nucleus 2.
## Z2 is Z of nucleus 2.
def deltaM(A1,Z1,A2,Z2):
    ## Compute difference in electron numbers e1 - e2.
    deltae = Z1 - Z2
    
    ## Get atomic masses.
    M1 = atomic_mass(A1,Z1)
    M2 = atomic_mass(A2,Z2)
    
    ## Compute nuclear mass difference (M1 - e1*me) - (M2 - e2*me) = M1 - M2 - me*(e1 - e2).
    deltaM = M1 - M2 - me*deltae

    return deltaM
##### end deltaM #####


##### begin electron_chemical_potential #####
# Computes electron chemical potential (MeV) from density, Ye, and temperature.
# Includes rest mass.
# electron density ne = int[0,inf]dE*g/( 1+exp((E-mue)/T) )
# positron density np = int[0,inf]dE*g/( 1+exp((E-mup)/T) )
# net lepton density = ne - np = rho * avogadro * Ye * 10**(-39)
# mue + mup + 2*me = 0
# rho = density (g/cm^3)
# Ye = electron fraction
# T = temperature (MeV)
def electron_chemical_potential(rhoYe,T):
    def equation(mu,n):
        leg = numpy.polynomial.legendre.leggauss(100)
        Plow = leg[0]    # E < mu
        weightlow = leg[1]

        lag = numpy.polynomial.laguerre.laggauss(100)
        Phigh = lag[0]    # E > mu
        weighthigh = lag[1]

        A = (math.pi**2 * hbar**3 * C**3)**(-1)*10**39    # MeV^-3 * cm^-3

        if mu > me:
            # If the total electron chemical potential is greater than the electron mass,
            # there is a shoulder in the distribution that should be integrated separately.
            P = (mu**2-me**2)**0.5 / 2 * (Plow+1)
            nelow = A * (mu**2-me**2)**0.5/2 * sum( weightlow * P**2 / \
                                                    ( 1 + numpy.exp(( (P**2+me**2)**0.5 - mu )/T) )    )
            P = Phigh + (mu**2-me**2)**0.5
            nehigh = A * sum( weighthigh * numpy.exp(Phigh) * P**2 / \
                              ( 1 + numpy.exp(( (P**2+me**2)**0.5 - mu )/T) )    )
            ne = nelow + nehigh
        else:
            # If there is no shoulder, don't integrate over it!
            P = Phigh
            ne = A * sum( weighthigh * numpy.exp(Phigh) * P**2 / ( 1 + numpy.exp(( (P**2+me**2)**0.5 - mu )/T) )    )

        P = Phigh
        np = A * sum( weighthigh * numpy.exp(Phigh) * P**2 / ( 1 + numpy.exp(( (P**2+me**2)**0.5 + mu )/T) )    )
        return ne-np-n

    # electron number density (cm^-3)
    n = rhoYe * avogadro
    return optimize.bisect(equation,0,100,args=(n,))
##### end electron_chemical_potential #####


##### begin electron_fermi_energy #####
# computes electron Fermi energy (MeV) from density and Ye
# does NOT include rest mass
# rho = density (g/cm^3)
# Ye = electron fraction
def electron_fermi_energy(rho,Ye):
    n = rho * avogadro * Ye * 10**(-39)    # electron number density (fm^-3)
    pF = hbar*C * (3*math.pi**2*n)**(1./3.)    # Fermi momentum (MeV)
    EF = (pF**2 + me**2)**(0.5) - me    # Fermi energy (MeV)
    return EF
##### end electron_fermi_energy #####


##### begin fortran_exp #####
# Takes a number to be formatted and a precision.
# Returns number formatted as 0.xEx
# input is number to be formatted.
# precision is number of significant digits.
def fortran_exp(input,precision):
    format_str = '{'+':.{}E'.format(precision-1)+'}'
    output = format_str.format(input)
    
    if not output.startswith('0'):
        output = output.split('E')
        output[0] = output[0].split('.')
        output[0][1] = output[0][1][:precision-1]
        output[0] = '0.{}{}'.format(*output[0])
        exp = int(output[1]) + 1
        output[1] = str(exp)
        if abs(exp) < 10:
            output[1] = output[1][:-1]+'0'+output[1][-1]
        if exp >= 0:
            output[1] = '+'+output[1]
        output = '{}E{}'.format(*output)
    else:
        format_str = '{'+':.{}E'.format(precision)+'}'
        output = format_str.format(input)
    
    return output
##### end fortran_exp #####


##### begin ft #####
# Computes log(ft) from transition strength B.
# B = the transition strength
# channel is the reaction channel
def ft(B,channel='gt'):
    if channel == 'gt':
        ft = B_to_ft/(avec_to_vec*B)
    
    if channel == 'f':
        ft = B_to_ft/B
        
    logft = numpy.log10(ft)
    
    print '\nlogft: {:.4}'.format(logft)

##### end ft #####


##### begin ft #####
# Computes log(ft) from transition strength B using FFN values of physical constants.
# B = the transition strength
# channel is the reaction channel
def ft_FFN(B,channel='gt'):
    if channel == 'gt':
        ft = 10**3.596/B
    
    if channel == 'f':
        ft = 10**3.791/B
        
    logft = numpy.log10(ft)
    
    print '\nlogft: {:.4}'.format(logft)

##### end ft #####


##### begin get_deltaM #####
# Computes effective mass differences M1 - M2 between nuclei in the same .lpt file.
# Lowest T = Tz states are the relative ground state energies.
# Energy difference between lowest T = Tz states is the effective mass difference.
# deltaM is returned as E1-E2 (MeV), where Ei is GS energy nucleus i.
# lpt_name is the path to the lpt file
# Tz1 in the z-component of isospin of nucleus 1
# Tz2 in the z-component of isospin of nucleus 2
def get_deltaM(lpt_name,Tz1,Tz2):
    ground1 = None
    ground2 = None
    
    lpt = open(lpt_name)
    inrange = False        
    for line in lpt:
        if inrange:    # If data header has been found, read in data
            if not line.isspace():      # skip blank lines
                line = line.split()
                E = float(line[6])
                try:
                    iso = float(line[3])
                except:
                    iso = line[3].split('/')
                    iso = float(iso[0])/float(iso[1])
                num = line[4]
                if '*' in num:
                    num = num[:-1]

                if iso >= abs(Tz1) and ground1 == None:    # true if state is ground state
                    ground1 = E
                if iso >= abs(Tz2) and ground2 == None:
                    ground2 = E
                    
                if ground1 != None and ground2 != None:
                    break
        else:           #If header not yet found, check if this line is it
            if re.search('#.*file.*j.*t.*#.*energy.*relative energy',line):
                inrange = True
    lpt.close()
    
    try:
        deltaM = ground1 - ground2
    except:
        print 'Error finding mass difference.'
        deltaM = None
        
    return deltaM
##### end get_deltaM #####


##### begin get_energy #####
# retrieves energy of given state from given lpt file
# returns None if state not found or ground state not found
# Tz is the z component of isospin
# J is the spin
# T is the isospin
# n is the number of the state with these properties (the first such state is n=1)
# lpt_name is the path to the lpt file if extracting from such file
# states[i] = [Ei,Ji,Ti,ni] 
def get_energy(Tz,J,T,n,lpt_name=None,states=None):
    if states:
        try:
            index = [s[1:] for s in states].index([J,T,n])
            E = states[index][0]
        except:
            print 'State not found when attempting to get energy.'
            E = None
    elif lpt_name:
        ground = None    # ground state energy initialized to None

        lpt = open(lpt_name)
        inrange = False        
        for line in lpt:
            if inrange:    # If data header has been found, read in data
                if not line.isspace():      # skip blank lines
                    line = line.split()
                    ener = float(line[6])
                    spin = line[2][0:-1]    # split spin from parity
                    try:
                        spin = float(spin)
                    except:
                        spin = spin.split('/')
                        spin = float(spin[0])/float(spin[1])
                    try:
                        iso = float(line[3])
                    except:
                        iso = line[3].split('/')
                        iso = float(iso[0])/float(iso[1])
                    if '*' in line[4]:
                        num = line[4][:-1]
                    else:
                        num = line[4]

                    if iso >= abs(Tz) and ground == None:    # true if state is ground state
                        ground = ener
                    if spin == J and iso == T and int(num) == int(n):    # true if state is desired state
                        E = ener - ground
                        break
            else:           #If header not yet found, check if this line is it
                if re.search('#.*file.*j.*t.*#.*energy.*relative energy',line):
                    inrange = True
        else:
            E = None
        lpt.close()
        
    return E
    
##### end get_energy #####


##### begin get_lpt #####
# searches given directory and returns .lpt file name
# will list all .lpt files found in given directory and ask user to choose one
# returns None if directory not found or no .lpt files found
def get_lpt(path='.',prompt='file'):
    lpt = []
    try:
        for f in os.listdir(path):
            if f.endswith('.lpt'):
                lpt.append(f)
    except:
        return None

    if len(lpt) == 0:
        lpt = None
    elif len(lpt) == 1:
        lpt = lpt[0]
    else:
        lpt = choose(prompt,lpt)

    return lpt
##### end get_lpt #####


##### begin get_lptstates #####
# Retrieve list of spins and energies from *.lpt
# Returns list of state energies and spins, isospins, and numbers.  states[i] = [Ei,Ji,Ti,ni]
# If no .lpt file found, returns None
# If no ground state found, returns -1
# lpt_path is filename for lpt file.  May or may not have '.lpt' suffix.
# Tz is z component of isospin.
def get_lptstates(lpt_path,Tz=None):
    try:
        lpt = open(lpt_path,'r')
    except:
        return None
        
    if Tz == None or Tz == 0:
        Tz = 0
        ground = 0
    else:
        ground = -1    # ground state energy initialized to -1 indicates that it must be found from Tz
        
    states = []    # states[i] = [Ei,Ji,Ti,ni]

    inrange = False
    for line in lpt:
        if inrange:    # If data header has been found, read in data
            if line.isspace()==False:      # skip blank lines
                line = line.split()
                ener = float(line[6])
                spin = line[2][0:-1]    # split spin from parity
                try:
                    spin = float(spin)
                except:
                    spin = spin.split('/')
                    spin = float(spin[0])/float(spin[1])
                try:
                    iso = float(line[3])
                except:
                    iso = line[3].split('/')
                    iso = float(iso[0])/float(iso[1])
                if '*' in line[4]:
                    num = line[4][:-1]
                else:
                    num = line[4]

                if iso >= abs(Tz) and ground < 0:    # find nucleus ground state energy
                    ground = ener

                if iso >= abs(Tz):    # only include states with large enough isospin to be in initial nucleus
                    states.append([ener-ground,spin,iso,num])    # subtract nucleus ground state energy
        else:           # If header not yet found, check if this line is it
            if re.search('#.*file.*j.*t.*#.*energy.*relative energy',line):
                inrange = True

    if ground < 0:
        states = None

    lpt.close()

    return(states)
##### end get_lptstates #####


##### begin get_lptexpstates #####
# retrieve list of spins and energies for measured states from *.lpt
# returns list of state energies and spins.  states[i] = [Ei,Ji,Ti,ni]
# if no .lpt file found, returns None
# if no ground state found, returns -1
# lpt_path is filename for lpt file.  May or may not have '.lpt' suffix.
# Tz is z component of isospin.
def get_lptexpstates(lpt_path,Tz=None):
    try:
        lpt = open(lpt_path,'r')
    except:
        return None
        
    if Tz == None or Tz == 0:
        Tz = 0
        ground = 0
    else:
        ground = -1    # ground state energy initialized to -1 indicates that it must be found from Tz
        
    states = []    # states[i] = [Ei,Ji,Ti,ni]

    inrange = False
    for line in lpt:
        if inrange:    # If data header has been found, read in data
            if line.isspace()==False:      # skip blank lines
                line = line.split()
                ener = float(line[6])
                spin = line[2][0:-1]    # split spin from parity
                try:
                    spin = float(spin)
                except:
                    spin = spin.split('/')
                    spin = float(spin[0])/float(spin[1])
                try:
                    iso = float(line[3])
                except:
                    iso = line[3].split('/')
                    iso = float(iso[0])/float(iso[1])
                if '*' in line[4]:
                    num = line[4][:-1]
                else:
                    num = line[4]

                if iso >= abs(Tz) and ground < 0:    # find nucleus ground state energy
                    ground = ener

                if line[0].startswith('e') and iso >= abs(Tz):    # only include states with large enough isospin to be in initial nucleus
                    states.append([ener-ground,spin,iso,num])    # subtract nucleus ground state energy
        else:           # If header not yet found, check if this line is it
            if re.search('#.*file.*j.*t.*#.*energy.*relative energy',line):
                inrange = True

    if ground < 0:
        return None

    lpt.close()

    return(states)
##### end get_lptexpstates #####


##### begin get_n #####
# retrieves n of the highest energy state less than or equal to
# specified energy in specified lpt file
# n is the number of the state with a given J, P, and T (the first such state is n=1)
# returns None if state not found or ground state not found
# Emax is the maximum energy
# Tz is the z component of isospin
# J is the spin
# P is the parity
# T is the isospin
# lpt_name is the path to the lpt file if extracting from such file
def get_n(E_max,Tz,J,P,T,lpt_name):
    if P == '0':
        P = '+'
    elif P == '1':
        P = '-'

    ground = -1    # ground state energy initialized to -1
    n = None    # n initialized to None

    lpt = open(lpt_name)
    inrange = False        
    for line in lpt:
        if inrange:    # if data header has been found, read in data
            if not line.isspace():      # skip blank lines
                line = line.split()
                
                ener = float(line[6])
                
                spin = line[2][0:-1]    # split spin from parity
                try:
                    spin = float(spin)
                except:
                    spin = spin.split('/')
                    spin = float(spin[0])/float(spin[1])
                
                par = line[2][-1]
                    
                try:
                    iso = float(line[3])
                except:
                    iso = line[3].split('/')
                    iso = float(iso[0])/float(iso[1])

                if iso >= abs(Tz) and ground < 0:    # true if state is ground state
                    ground = ener
                    
                if ground >= 0 and spin == J and par == P and iso == T:
                    E = ener - ground
                    if E > E_max:
                        break
                    
                    if '*' in line[4]:
                        n = line[4][:-1]
                    else:
                        n = line[4]
                    
        else:           #If header not yet found, check if this line is it
            if re.search('#.*file.*j.*t.*#.*energy.*relative energy',line):
                inrange = True
        
    lpt.close()
    
    return n
##### end get_n #####


##### begin get_strength #####
# Retrieves strength for specified channel in specified nucleus.
# Strength is returned as strength[i] = [Ei,Ji,Ti,ni,[[deltaEi1,Bi1],...]].
# If requested, Fermi strength will be included.
# AZ is 'A,Z' of nucleus.
# channel is the reaction channel.
# quench is the quenching factor.
# path is path to overlaps.
# fermi indicates whether to include Fermi strength.
# lptf_name is the final nucleus lpt file.  Necessary if Fermi strength is desired.
# hist indicates whether to return a weighted histogram.
def get_strength(AZ,channel,quench=quench,path='overlaps/',hist=None):
    if not path.endswith('/'):
        path = path+'/'
        
    strength = []    # strength[i] = [Ei,Ji,Ti,ni,[[deltaEi1,Bi1],...]]
    
    for dao in os.listdir(path):
        if os.path.splitext(dao)[1] == '.dao':
            info = os.path.splitext(dao)[0].split()
        else:
            continue
        
        ### upper is for old format, lower is for new; comment/uncomment as appropriate
        #if info[0] == channel and info[1].split('=')[1] == AZ.split(',')[1]:
        if info[0].split('=')[1] == AZ and info[1] == channel:
            E = float(info[2].split('=')[1])
            J = float(info[3].split('=')[1])
            T = float(info[4].split('=')[1])
            n = info[5].split('=')[1]
            
            strength.append([E,J,T,n,[]])
            
            f = open(path+dao)
            
            if channel == 'gt+' or channel == 'gt-':
                # skip header
                f.readline()
            
                for line in f:
                    line = line.split()
                    deltaE = float(line[0])
                    if line[1].startswith('e') or line[1].startswith('m') or line[1].startswith('g'):
                        B = float(line[1][1:])
                    else:
                        B = quench*float(line[1])
                    if B > 0:
                        strength[-1][-1].append([deltaE,B])
            elif channel == 'gt3':
                for line in f:
                    line = line.split()
                    deltaE = float(line[1])
                    if line[2].startswith('e') or line[2].startswith('m'):
                        B = float(line[2][1:])
                    else:
                        B = quench*float(line[2])
                    ### upper is for old format, lower is for new; comment/uncomment as appropriate
                    #if B > 0 and deltaE < 25:
                    if B > 0:
                        strength[-1][-1].append([deltaE,B])
                        
    return strength
##### end get_strength #####


##### begin get_strengthF #####
# Retrieves Fermi strength for specified channel in specified nucleus.
# Only retrieves strength for which other overlaps have been computed.
# Strength is returned as strength[i] = [Ei,Ji,[[deltaEi1,Bi1],...]].
# AZi is 'A,Z' of initial nucleus
# channel is the reaction channel
# path is path to overlaps
def get_strengthF(AZi,channel,lptf_name,path='overlaps/'):
    # return nothing if not a charged current channel
    if not (channel == 'gt-' or channel == 'gt+'):
        return []
        
    if not path.endswith('/'):
        path = path+'/'
    strength = []    # strength[i] = [Ei,Ji,Ti,ni,[[Qi1,Bi1],...]]
    
    A = int(AZi.split(',')[0])
    Zi = int(AZi.split(',')[1])
    Tzi = (2.0*Zi - A)/2.0
    
    if channel == 'gt-':
        Zf = Zi - 1
        Tzf = Tzi - 1
    elif channel == 'gt+':
        Zf = Zi + 1
        Tzf = Tzi + 1
    
    # dM = Mf - Mi
    dM = deltaM(A,Zf,A,Zi)
    
    for dao in os.listdir(path):
        if os.path.splitext(dao)[1] == '.dao':
            info = os.path.splitext(dao)[0].split()
        else:
            continue
        
        # upper is for old format, lower is for new; comment/uncomment as appropriate
        #if info[0] == channel and info[1].split('=')[1] == AZi.split(',')[1]:
        if info[0].split('=')[1] == AZi and info[1] == channel:
            Ei = float(info[2].split('=')[1])
            J = float(info[3].split('=')[1])
            T = float(info[4].split('=')[1])
            n = info[5].split('=')[1]
            
            if ((channel == 'gt+' and Tzi < 0) or (channel == 'gt-' and Tzi > 0)) or T > abs(Tzi):
                strength.append([Ei,J,T,n,[]])
                
                Ef = get_energy(Tzf,J,T,n,lpt_name=lptf_name)
                
                Q = Ef - Ei + dM
                
                if channel == 'gt-':
                    BF = T*(T+1) - Tzi*(Tzi-1)
                elif channel == 'gt+':
                    BF = T*(T+1) - Tzi*(Tzi+1)
                strength[-1][-1].append([Q,BF])
    
    return strength
##### end get_strength #####


##### begin histogram #####
# Computes weighted histogram of input array.
# X is array to be binned
# data is array of x-values and weights: data[i] = [xi,wi]
# bins is array of bin edges
def histogram(data,bins):
    hist = numpy.zeros(len(bins)-1)
    
    for d in data:
        if bins[0] <= d[0] <= bins[-1]:
            for b in range(len(bins)):
                if d[0] < bins[b]:
                    hist[b-1] = hist[b-1] + d[1]
                    break
    
    return hist
##### end histogram #####


##### begin imbalance #####
# Computes imbalance between inputs.
# A, B are the quantities between which to compute imbalance.  May be numbers or arrays.
def imbalance(A,B):
    imbalance = (A - B)/(A + B)
    
    return imbalance
##### end imbalance #####


##### begin kernel_cc #####
## Computes and returns charged lepton capture/decay phase space integral kernel.
## A (int) is parent nucleus mass number.
## Z (int) is parent nucleus proton number.
## w (float, (array)) is charged lepton energy (units of electron mass).
## q (float) is transition energy (units of electron mass).
## T (float, (array)) is temperature (units of electron mass).
## mu (float, (array)) is charged lepton Fermi energy (units of electron mass).  Must have same array shape as T.
## lepton (str) is charged lepton in interaction ('electron' or 'positron').
## capdec (str) indicates the interaction ('capture' or 'decay').
def kernel_cc(A,Z,w,q,T,mu,lepton,capdec):
    ## Make sure T and mu are appropriately shaped arrays.
    T = numpy.array([T])
    T.shape = (T.size,1)
    mu = numpy.array([mu])
    mu.shape = (mu.size,1)
    
    ## Get neutrino energy, charged lepton availability (blocking) factor, and daughter nucleus charge (if decay).
    if capdec == 'capture':
        w_nu = w - q
    elif capdec == 'decay':
        w_nu = -w - q
        if lepton == 'electron':
            Z = Z+1
        elif lepton == 'positron':
            Z = Z-1
        
    ## Charged lepton momentum (mc).
    p = numpy.sqrt(w**2-1.)
    
    ## Set unphysical charged lepton and neutrino energies to zero.
    ## This will make them not contribute to K.
    #for e in range(len(w)):
    #    if w[e] < 1:
    #        w[e] = 1.
    #    if w_nu[e] < 0:
    #        w_nu[e] = 0.
    
    ## Nuclear radius.
    R = nuc_rad(A)
    
    ## s in FFNI eqn 5b.
    ffn_s = numpy.sqrt( (1-(alpha*Z)**2) )
    
    ## F is the Coulomb correction.
    ## alpha*Z*w/p is eta in FFNI.
    ## nan_to_num on exponential necessary for values of w near 1.
    ## exponential must be at the end to ensure it's never multiplied by a value greater than 1.
    if lepton == 'electron':
        polynomial = 2*(1+ffn_s) * (2*p*R)**(2*(ffn_s-1))
        exponential = numpy.nan_to_num( numpy.exp(pi*alpha*Z*w/p) )
        gammas = abs( scipy.special.gamma(ffn_s+alpha*Z*w/p*1j) / scipy.special.gamma(2*ffn_s+1) )**2
        F = polynomial * (gammas * exponential)
    elif lepton == 'positron':
        polynomial = 2*(1+ffn_s) * (2*p*R)**(2*(ffn_s-1))
        exponential = numpy.nan_to_num(numpy.exp(pi*(-alpha*Z*w/p)))
        gammas = abs( scipy.special.gamma(ffn_s+(-alpha*Z*w/p)*1j) / scipy.special.gamma(2*ffn_s+1) )**2
        F = polynomial * (gammas * exponential)
        
    ## Charged lepton occupation probability.
    fL = 1/( 1+numpy.exp((w-mu)/T) )
    
    ## Finally, the kernel.
    ## K[t][e] is the value of the kernel at temperature T[t] and electron energy w[e].
    ## 1/(1+numpy.exp( (w-mu)/T )) is the charged lepton occupation probability.
    if capdec == 'capture':
        K = fL * F * w * p * w_nu**2
    elif capdec == 'decay':
        K = ( 1.0 - fL ) * F * w * p * w_nu**2
        
    K = numpy.nan_to_num(K)
    
    ## K[t][e] is the value of the kernel at temperature T[t] and electron energy w[e].
    return(K)
        
##### end kernel_cc #####


##### begin logft #####
# Computes log(ft) value for a known transition with a known lifetime.
# A (int) is nuclear mass number.
# Z (int) is proton number.
# Q (float) is transition energy.
# halflife (float) is the measured half life of the transition in seconds.
# lepton is the outgoing charged lepton.
# F is the Coulomb correction.  If not specified, treat nucleus as fully ionized.
def logft(A,Z,Q,halflife,lepton,F=None):
    ##### begin kernel: lepton decay phase space kernel #####
    # w (float) is lepton energy (units of electron mass)
    # q (float) is transition energy (units of electron mass)
    # lepton is charged lepton in interaction ('electron' or 'positron')
    def kernel(w,q,lepton,F):
        print '\nw:'
        print w
        # charged lepton momentum (mc)
        p = (w**2-1.)**(0.5)
        
        # neutrino energy
        w_nu = -w - q
    
        if lepton == 'electron':
            eta = alpha*Z*w/p
        elif lepton == 'positron':
            eta = -alpha*Z*w/p
    
        # s in FFNI eqn 5b
        ffn_s = (1-(alpha*Z)**2)**(0.5)
    
        # Coulomb correction
        if not F:
            F = 2*(1+ffn_s) * (2*p*R)**(2*(ffn_s-1)) * numpy.exp(pi*eta) * abs(scipy.special.gamma(ffn_s+eta*1j)/scipy.special.gamma(2*ffn_s+1))**2

        H = F*w*p*w_nu**2

        return(H)
    ##### end kernel #####
    
    # nuclear radius
    R = nuc_rad(A)
    
    # adjust Z to daughter nucleus value
    if lepton == 'electron':
        Z = Z+1
    elif lepton == 'positron':
        Z = Z-1
    
    # convert to units of electron mass
    q = Q/me
    
    #[[sample points],[weights]] for Gauss-Legendre integration
    leg = numpy.polynomial.legendre.leggauss(100)
    points = leg[0]
    weights = leg[1]
    print '\npoints:'
    print points
    print '\nweights'
    print weights
    
    # conversions to make limits of integration -1 to 1
    X = (-q - 1.)/2.
    Y = (-q + 1.)/2.
    
    # phase space integral
    phase_int = sum(weights*X*kernel(X*points+Y,q,lepton,F))
    
    logft = numpy.log10(halflife*phase_int)

    return logft
##### end logft #####


##### begin nuc_rad #####
# Computes nuclear radius from mass number.
# A is nuclear mass number.
def nuc_rad(A):
    R = 2.908*10**(-3)*A**(1./3.)-2.437*10**(-3)*A**(-1./3.)
    
    return R
##### end nuc_rad #####


##### begin pair_erate #####
# computes energy loss rate per baryon from nuclei via neutrino pairs
# deltaE is the energy difference between the initial and final states (MeV)
# BGT is squared reduced matrix element for transition
# quench is the quenching factor.  Pre-quenched BGT should be passed with default quench.
# returns rates in MeV/s
def pair_erate(deltaE,BGT,quench=1):
    rate = 1.71*10**(-4)*quench*BGT*(deltaE)**6

    return rate
##### end pair_erate #####
    

##### begin partfcn #####
# computes the partition function from temperature T and list of states or an lpt file
# T is temperature in MeV.  May be scalar or array.
# states[i] = [Ei,Ji,<whatever else>]
# lpt_name is file name for lpt.
# Tz is isospin.  May be neglected if lpt file contains no states with bad isospins
# returns partition function G(T)
def partfcn(T,states=None,lpt_name=None,Tz=None):
    G = 0
    if not states:
        states = get_lptstates(lpt_name,Tz)
        
    for s in states:
        E = s[0]
        J = s[1]
        G = G + (2*J+1)*numpy.exp(-E/T)

    return(G)
##### end partfcn #####


##### begin partfct_partial #####
# Computes partition function from temperature T
# and an abridged set of states from an lpt file.
# This is tantamount to finding the total thermodynamic weight of those states.
# T is the temperature.  May be scalar or array.
# lpt_name is file name for lpt.
# E_min and E_max set the energy range of accepted states.
# Tz is isospin.  May be neglected if lpt file contains no states with bad isospins
# returns partial partition function Gp(T)
def partfcn_partial(T,states=None,lpt_name=None,E_min=0,E_max=float('inf'),Tz=None):
    Gp = 0
    if not states:
        states = get_lptstates(lpt_name,Tz)
    
    for s in states:
        E = s[0]
        J = s[1]
        if E_min <= E < E_max:
            Gp = Gp + (2*J+1)*numpy.exp(-E/T)
    
    return(Gp)
##### end partfcn_partial #####


##### begin rates_cc #####
## Charged lepton compute capture rate and energy loss rate from Ef and strength distribution.
## A is parent nucleus mass number.
## Z is parent nucleus proton number.
## strength[t][q] = 1/ft at temperature[t] and Q[q] where Q is in MeV.
## Q is array of Q-values associated with strength (MeV).
## T = temperature (MeV).
## mue = electron Fermi energy (MeV).
## lepton is charged lepton.
## capdec is whether charged lepton is captured or emitted.
## shoulderpoints is number of points in Legendre-Gauss quadrature.
## tailpoints is number of points in Gauss-Laguerre quadrature.
def rates_cc(A,Z,strength,T,Q,mue,lepton,capdec,shoulderpoints=64,tailpoints=64):
    ## Make sure T and mue are arrays.
    T = numpy.array([T])
    T.shape = T.size
    mue = numpy.array([mue])
    mue.shape = mue.size
    
    ## Express T, Q, and mue in units of electron mass.
    T = T/me
    Q = Q/me
    mue = mue/me
    
    ## Set sign of charged lepton chemical potential.
    if lepton == 'electron':
        mu = mue
    elif lepton == 'positron':
        mu = -mue
    
    ## Gauss-Legendre info for integrating shoulder.
    ws,weights = numpy.polynomial.legendre.leggauss(shoulderpoints)
    
    ## Gauss-Laguerre info for integrating tail.
    wt,weightt = numpy.polynomial.laguerre.laggauss(tailpoints)
    
    ## Lepton capture/decay rate in events per second.
    reaction_rate = numpy.zeros(len(T))
    
    ## Energy lost via neutrinos in MeV.
    energy_loss_rate = numpy.zeros(len(T))
    
    for t in range(len(T)):
        for q in range(len(Q)):
            ## 1/ft.
            ft = strength[t][q]
            
            ## Check whether this transition is unphysical or irrelevant.
            if (capdec == 'decay' and Q[q] > -1) or ft == 0.:
                continue
                
            ## Set overall limits of integration.
            if capdec == 'capture':
                lower_lim = max(1,Q[q])    ## 1 is for first term, Q[q] for second term of MFB(2014) eqn 14.
                upper_lim = numpy.inf
            elif capdec == 'decay':
                lower_lim = lower_lim = 1
                upper_lim = -Q[q]
            
            ## Phase space integral and energy loss rate for this transition at this temperature.
            phase_int = 0
            loss = 0
            
            ## Integrate shoulder.
            ## mu[t] > lower_lim implies a shoulder in lepton distribution that can be integrated separately, and leptons in the shoulder participate.
            if mu[t] > lower_lim:
                ## X and Y are change of variables for Gauss-Legendre integration of shoulder.
                ## X scales and Y displaces to set integration limits to (-1,1).
                X = ( min(mu[t],upper_lim) - lower_lim )/2.
                Y = ( min(mu[t],upper_lim) + lower_lim )/2.
                w = ws*X+Y
                ## shoulder is an array of the terms in the Gauss-Legendre quadrature sum.
                shoulder = weights * X * kernel_cc(A, Z, w, Q[q], T[t], mu[t], lepton, capdec)[0]
                phase_int = phase_int + sum(shoulder)
                if capdec == 'capture':
                    loss = loss + sum( shoulder*(w - Q[q]) )
                elif capdec == 'decay':
                    loss = loss + sum( shoulder*(-w - Q[q]) )
                    
            ## Integrate tail.
            if capdec == 'capture':
                ## For capture, limits of integration are ( max(lower_lim,mu[t]), inf ).
                ## mu[t] is for when shoulder is integrated separately.
                ## Change of variables for Gauss-Laguerre quadrature: w = T[t]*wt + lower_lim.
                ## Displacement sets lower integration limit to zero, allowing Gauss-Laguerre quadrature.
                ## Rescaling by the temperature T[t] stretches the charged lepton thermal tail to be appropriately sampled by the quadrature.
                ## Rescaling implies a factor of T[t] out front.
                w = T[t]*wt + max(lower_lim, mu[t])
                ## tail is an array of the terms in the Gauss-Laguerre quadrature sum.
                tail = weightt * numpy.exp(wt) * T[t] * kernel_cc(A, Z, w, Q[q], T[t], mu[t], lepton, capdec)[0]
                #tail = weightt*numpy.exp(wt)*kernel_cc(A,Z,wt+max(1,Q[q]),Q[q],T,mu,lepton,capdec)
                phase_int = phase_int + sum(tail)
                loss = loss + sum( tail*(w - Q[q]) )
            elif capdec == 'decay':
                ## Since the upper limit is finite, we will reuse Gauss-Legendre quadrature.
                ## Gauss-Laguerre won't work because charged lepton energies greater than -Q will yield neutrinos with negative energy.
                ## I don't like this, but I don't know another quadrature method.  Consider learning one?
                ## If outgoing charged lepton cannot go into tail, don't bother with the tail.
                if upper_lim <= mu[t]:
                    continue
                ## Limits on tail part are ( max(lower_lim,mu[t]), upper_lim )
                X = ( upper_lim - max(lower_lim,mu[t]) )/2.
                Y = ( upper_lim + max(lower_lim,mu[t]) )/2.
                w = ws*X+Y
                ## tail is an array of the terms in the Gauss-Legendre quadrature sum.
                tail = weights * X * kernel_cc(A, Z, w, Q[q], T[t], mu[t], lepton, capdec)[0]
                phase_int = phase_int + sum(tail)
                loss = loss + sum( tail*(-w - Q[q]) )
            
            reaction_rate[t] = reaction_rate[t] + numpy.log(2) * ft * phase_int
            energy_loss_rate[t] = energy_loss_rate[t] + numpy.log(2) * ft * loss * me
            
    return numpy.array([reaction_rate,energy_loss_rate]).transpose()
##### end rates_cc #####


##### begin sci_not #####
# Converts input to a string formatted as scientific notation in LaTeX.
# This includes putting a superscript in the exponent.
# input is number to be converted.  Can be anything that casts to float.
# precision indicates how many decimal places to include in the coefficient.
def sci_not(input,precision=2):
    try:
        input = float(input)
    except:
        print 'Cannot cast input to float in weaklib.sci_not().'
        return None
    precision = str(int(precision))
    
    # Create format specification string
    formatstring = '{:.'+precision+'e}'
    
    # Format input according to specification
    output = formatstring.format(input)
    
    # Split output into coefficient and exponent
    output = output.split('e')
    output[1] = str(int(output[1]))
    
    # Combine coefficient and exponent
    output = output[0]+'$\\times$10$^{'+output[1]+'}$'
    
    return output
##### end sci_not #####


##### begin spectrum_cc #####
# Nuclear neutrino spectrum from charged current interactions in neutrinos/nucleus/s/MeV.
# Charged lepton compute capture rate and energy loss rate from Ef and strength distribution.
# A is parent nucleus mass number.
# Z is parent nucleus proton number.
# strength[i] = [Qi,1/ft(if)]     Q is in MeV, ft(if) is the ft value.
# T (float, list, or array) = temperature (MeV)
# mue = electron Fermi energy (MeV)
# lepton is charged lepton.
# capdec is whether charged lepton is captured or emitted.
# Enu (array) is neutrino energies at which to compute spectral density.
def spectrum_cc(A,Z,strength,T,mue,lepton,capdec,Enu):
    # Make sure T and mu are appropriately shaped arrays.
    T = numpy.array([T])
    T.shape = (T.size,1)
    mue = numpy.array([mue])
    mue.shape = (mue.size,1)
    
    if lepton == 'electron':
        mu = mue
    elif lepton == 'positron':
        mu = -mue
    
    mu = mu/me    # express mu in units of electron mass
        
    spect = numpy.zeros((len(T),len(Enu)))

    for s in strength:
        q = s[0]/me    # express Q in units of electron mass
        
        # Get charged lepton energies.
        if capdec == 'capture':
            w = Enu/me + q    # incoming charged lepton energy for an outgoing neutrino of energy Enu[e] in units of electron mass
        elif capdec == 'decay':
            w = -Enu/me - q    # outgoing charged lepton energy for an outgoing neutrino of energy Enu[e] in units of electron mass
        
        # Set unphysical charged lepton energies to 1.
        # In kernel_CC(), this will give a momentum of 0 and thus zero contribution at the corresponding neutrino energy.
        for e in range(len(w)):
            if w[e] < 1.0:
                w[e] = 1.0
        
        spect = spect + numpy.log(2)*s[1]*kernel_cc(A,Z,w,q,T,mu,lepton,capdec)/me
        
    return (spect)
##### end spectrum_cc #####


##### begin spectrum_nc #####
# Computes (anti)neutrino spectrum from pair creation in neutrinos/nucleus/s/MeV.
# Includes all flavors.
# strength[i] = [Qi,BGT3]     Q is in MeV, BGT3 is the strength.
# Enu (array) is neutrino energies at which to compute spectral density.
def spectrum_nc(strength,Enu):
    spect = numpy.zeros(len(Enu))

    for s in strength:
        Q = s[0]
        BGT = s[1]
        
        if Q < 0 and BGT > 0:
            E = numpy.array([e for e in Enu if e < -Q])    # only use physical values of Enu
            E = numpy.pad(E,(0,len(Enu)-len(E)),'constant',constant_values=0.)    # pad array to the same length as Enu
            spect = spect + (G_F**2*avec_to_vec/(2*pi**3*hbar))*BGT*E**2*(-Q-E)**2
        
    return (spect)
    
#    E = spectrum[s][0]
#    if E > deltaE:
#        break
#    rate = weight*E**2*(deltaE-E)**2
#    spectrum[s][1] = spectrum[s][1] + rate
    
#    return spectrum

##### end spectrum_nc #####


##### begin spectrum_nc #####
# Computes (anti)neutrino spectrum from pair creation.
# deltaE is nuclear transition energy
# weight is numerical factor that includes strength and thermal weight
# step is the energy step size in the spectrum (MeV)
# spectrum is a neutrino spectrum to be added to.
#def spectrum_nc(deltaE,weight=1,step=0.1,spectrum=None):
#    deltaE = abs(deltaE)
#    if not spectrum:
#        spectrum = numpy.arange(0,deltaE+step,step).tolist()    # spectrum[i] = [Ei,ratei]
#        for s in range(len(spectrum)):
#            spectrum[s] = [spectrum[s],0.]
            
#    for s in range(len(spectrum)):
#        E = spectrum[s][0]
#        if E > deltaE:
#            break
#        rate = weight*E**2*(deltaE-E)**2
#        spectrum[s][1] = spectrum[s][1] + rate
    
#    return spectrum

##### end spectrum_nc #####


##### begin thermal_strength #####
## Constructs and returns binned thermal strength distribution for specified nucleus at specified temperature.
## thermal_strength[t][e] = strength at temperature t for Q-value e
## Uses modified Brink hypothesis.
## AZ is 'mass,charge' of nucleus.
## channel is reaction channel.
## temp is temperature (MeV).
## lpt is path to .lpt file for nucleus.
## path is path to overlap files.
## quench is quenching factor for strength.
## cutoff1 is maximum initial state energy for exact states in modified Brink (MeV).
## cutoff2 is lower edge of highest initial state energy bin.
## bwidth is width of high energy average state bins in modified Brink (MeV).
## dEmax is maximum included Q-value.
## dEwidth is width of Q-value bins in thermal strength distribution (MeV).
def thermal_strength(AZ,channel,temp,overlappath,format,lptpath='./',quench=quench,cutoff1=15.,cutoff2=20.,bwidth=1.,dEmax=50.,dEwidth=0.1,spam=True):
    ## Get A and Z, compute Tz.
    A,Z = numpy.fromstring(AZ,sep=',',dtype=int)
    Tz = (2.*Z-A)/2.
        
    ## Make sure paths end with a '/'.
    if not lptpath.endswith('/'):
        lptpath = lptpath+'/'
    if not overlappath.endswith('/'):
        overlappath = overlappath+'/'
        
    ## Generate lists of all parent states used in computation, sorted by energy.
    states = []    ## states[i] = [Ei,Ji,Ti,ni,dao path i]
    if format == 'oxbash':
        ## Select appropriate lpt files.  For gt- and gt+, we need daughter nuclei to get Fermi transition energies.
        lpt = None
        lptm = None
        lptp = None
        for f in os.listdir(lptpath):
            if f.startswith('A,Z=') and f.endswith('.lpt'):
                az = f.split()[0].split('=')[1]
                a,z = numpy.fromstring(az,sep=',',dtype=int)
                ## If this is the lpt of the initial nucleus, set it.
                if a == A and z == Z:
                    lpt = lptpath + f
                ## If this is the lpt for a daughter nucleus, set it.
                if channel == 'gt-' and a == A and z == Z-1:
                    lptm = lptpath + f
                if channel == 'gt+' and a == A and z == Z+1:
                    lptp = lptpath + f
        if not lpt:
            print 'Parent nucleus lpt file not found.'
            exit()
        if (not lpt) or (channel == 'gt-' and not lptm) or (channel == 'gt+' and not lptp):
            print '{} daughter nucleus lpt file not found.'.format(channel)
            return None
        
        for dao in os.listdir(overlappath):
            if os.path.splitext(dao)[1] == '.dao':
                info = os.path.splitext(dao)[0].split()
                if info[0].split('=')[1]==AZ and info[1]==channel:
                    E = float(info[2].split('=')[1])
                    J = float(info[3].split('=')[1])
                    T = float(info[4].split('=')[1])
                    n = int(info[5].split('=')[1])
                
                    if E < cutoff2+bwidth:
                        states.append([E,J,T,n,overlappath+dao])
        states.sort(key=lambda s: s[0])
    elif format == 'bigstick':
        ## Get names of nuclei and reaction channels.
        nuc_name = atomic_names.get(Z)
        if channel == 'gt-':
            chan_name = 'Bplus'
        elif channel == 'gt+':
            chan_name = 'Bminus'
        try:
            ## Need a flag for when the reverse reaction is the one we have computed strengths for.
            forward = True
            with open(overlappath+'{}{}{}_expt.gtr'.format(nuc_name,A,chan_name)) as f:
                data = f.readlines()
        except IOError:
            ## If parent strength file not found, check for the daughter.
            print '\nOverlap file {} not found.\nChecking reverse reaction.'.format(overlappath+'{}{}{}_expt.gtr'.format(nuc_name,A,chan_name))
            if channel == 'gt-':
                nuc_name = atomic_names.get(Z-1)
                chan_name = 'Bminus'
            elif channel == 'gt+':
                nuc_name = atomic_names.get(Z+1)
                chan_name = 'Bplus'
            try:
                forward = False
                with open(overlappath+'{}{}{}_expt.gtr'.format(nuc_name,A,chan_name)) as f:
                    data = f.readlines()
            except IOError:
                print 'Overlap file {} also not found.\n'.format(overlappath+'{}{}{}_expt.gtr'.format(nuc_name,A,chan_name))
                return None
            
        ## Get number of parent states.
        count_p = int(float(data[5]))
        ## Read in parent state info.
        for c in range(count_p):
            info = data[c+7].split()
            n = int(info[0])
            if n != c+1:
                print 'Error reading parent states.\nExpected state number {}, got state number {}.'.format(p+1,n)
                exit()
            J = float(info[1])
            T = float(info[2])
            E = float(info[3])
            if E < cutoff2+bwidth:
                states.append([E,J,T,n])
        states.sort(key=lambda s: s[3])
        
        ## Get number of daughter states.
        count_d = int(float(data[7 + count_p + 3]))
        ## Read in daughter state info.
        states_d = []
        for c in range(count_d):
            info = data[c + 7 + count_p + 4].split()
            n = int(info[0])
            if n != c+1:
                print 'Error reading daughter states.\nExpected state number {}, got state number {}.'.format(p+1,n)
                exit()
            J = float(info[1])
            T = float(info[2])
            E = float(info[3])
            states_d.append([E,J,T,n])
        states_d.sort(key=lambda s: s[3])
        
        ## If reading daughter nucleus strength file, swap lists of parent and daughter states.
        if not forward:
            states,states_d = states_d,states

    ## Create hea_bins to average high energy states where not all overlaps are computed.
    hea_bins = numpy.arange(cutoff1,cutoff2,bwidth).tolist()    ## hea_bins[i] = [lower bin edge, number of included parent states weighted by spin]
    if hea_bins[-1] < cutoff2:
        hea_bins.append(cutoff2)
    ## Add slots to store total spin degeneracy.
    for e in range(len(hea_bins)):
        hea_bins[e] = [hea_bins[e],0.]

    ## Compute total spin degeneracy of binned hea states.
    for s in states:
        E = s[0]
        J = s[1]
        if E < cutoff1:
            continue
        for e in range(len(hea_bins)):
            if hea_bins[e][0] <= E < hea_bins[e][0]+bwidth:
                hea_bins[e][1] = hea_bins[e][1] + (2.*J+1.)
                break
        else:
            print 'Error in weaklib.thermal_strength() computing hea state spin degeneracy.'
    
    ## Get daughter - parent nuclear mass difference and compute dEmin.
    if channel == 'gt+':
        dM = deltaM(A,Z+1,A,Z)
    elif channel == 'gt-':
        dM = deltaM(A,Z-1,A,Z)
    elif channel == 'gt3':
        dM = 0.
    dEmin = dM - (cutoff2 + bwidth)
    ## Round dEmin to lowest multiple of dEwidth.  Outer round is to deal with floating point errors.
    dEmin = round(dEwidth * math.floor(dEmin/dEwidth),10)
    
    ## Create transition energy bins, round off to avoid floating point errors.
    dE = numpy.arange(dEmin,dEmax+dEwidth,dEwidth)
    for e in range(len(dE)):
        dE[e] = round(dEwidth * math.floor(dE[e]/dEwidth),10)
    
    ## Compute partition function.
    if format == 'oxbash':
        G = partfcn(temp,lpt_name=lpt,Tz=Tz)
    elif format == 'bigstick':
        G = partfcn(temp,states=states)
    
    ## Compute partial partition functions for high energy average states.
    hea_G = []    ## hea_G[e][t] = partial partition function for hea_bin[e] at temp[t]
    for e in hea_bins:
        E_min = e[0]
        E_max = e[0] + bwidth
        if format == 'oxbash':
            hea_G.append(partfcn_partial(temp,lpt_name=lpt,E_min=E_min,E_max=E_max,Tz=Tz))
        elif format == 'bigstick':
            hea_G.append(partfcn_partial(temp,states=states,E_min=E_min,E_max=E_max))
    hea_G = numpy.array(hea_G)
    
    ## Extract strength of parent states, bin them, and add to total thermal strength.
    thermal_strength = numpy.zeros((len(temp),len(dE)))    ## thermal_strength[t][e] = B(Q) at temp[t], deltaE[e]
    if format == 'oxbash':
        counter = 0
        for s in states:
            counter = counter + 1
            if spam:
                print 'Computing thermal strength in {} of {} states.'.format(counter,len(states))
            E = s[0]
            J = s[1]
            T = s[2]
            n = s[3]
            dao = s[4]
            
            ## Compute statistical weight of state.
            if E < cutoff1:
                g = (2.*J+1.)*numpy.exp(-E/temp)/G
            else:
                for e in range(len(hea_bins)):
                    if hea_bins[e][0] <= E < hea_bins[e][0]+bwidth:
                        g = ((2.*J+1.)/hea_bins[e][1])*(hea_G[e]/G)
                        break
                else:
                    print 'Error in assigning statistical weight to high energy state {}.'.format(dao)
                    raw_input('pause')
            
            ## Load the overlap data and add to thermal_strength.
            f = open(dao)
            
            ## Skip header.
            if channel == 'gt+' or channel == 'gt-':
                f.readline()
            
            ## Read in deltaE and B values.
            for line in f:
                line = line.split()
                if channel == 'gt+' or channel == 'gt-':
                    deltaE = float(line[0])
                    B = line[1]
                elif channel == 'gt3':
                    deltaE = float(line[1])
                    B = line[2]
                ## Round deltaE to nearest multiple of dEwidth.
                deltaE = round(dEwidth * round(deltaE/dEwidth),10)
                ## If experimental, mirror, or guessed strength, don't quench.
                if B.startswith('e') or B.startswith('m') or B.startswith('g'):
                    B = float(B[1:])
                else:
                    B = quench * float(B)
                ## Figure out dE bin of this transition and add to thermal_strength.
                if B > 0 and deltaE < dEmax:
                    if channel == 'gt-' or channel == 'gt+':
                        thermal_strength[:,numpy.where(dE==deltaE)[0][0]] = thermal_strength[:,numpy.where(dE==deltaE)[0][0]] + g*B*avec_to_vec
                    elif channel == 'gt3':
                        thermal_strength[:,numpy.where(dE==deltaE)[0][0]] = thermal_strength[:,numpy.where(dE==deltaE)[0][0]] + g*B
                
            ## For charged current, include Fermi strength if parent is proton/neutron rich (Tz >/< 0)
            ## or if total isospin of initial state is greater than |Tz| of initial state (T > abs(Tz))
            BF = 0
            if channel == 'gt-' and (Tz > 0 or T > abs(Tz)):
                Ef = get_energy(Tz-1,J,T,n,lpt_name=lptm)
                deltaE = Ef - E + dM
                BF = T*(T+1) - Tz*(Tz-1)
            if channel == 'gt+' and (Tz < 0 or T > abs(Tz)):
                Ef = get_energy(Tz+1,J,T,n,lpt_name=lptp)
                deltaE = Ef - E + dM
                BF = T*(T+1) - Tz*(Tz+1)
            ## Round deltaE to nearest multiple of dEwidth.
            ## Figure out dE bin of this transition and add to thermal_strength.
            deltaE = round(dEwidth * round(deltaE/dEwidth),10)
            if BF > 0 and deltaE < dEmax:
                thermal_strength[:,numpy.where(dE==deltaE)[0][0]] = thermal_strength[:,numpy.where(dE==deltaE)[0][0]] + g*BF
    elif format == 'bigstick':
        ## For each transition, add its strength to thermal_strength.
        for t in range(int(float(data[7 + len(states) + 4 + len(states_d) + 1]))):
            info = data[7 + len(states) + 4 + len(states_d) + 3 + t].split()
            ## Get parent state energy and spin.
            if forward:
                np = int(info[0])
            else:
                np = int(info[1])
            Ep = states[np-1][0]
            Jp = states[np-1][1]
            ## Compute statistical weight of state.
            if Ep < cutoff1:
                g = (2.*Jp+1.)*numpy.exp(-Ep/temp)/G
            else:
                for e in range(len(hea_bins)):
                    if hea_bins[e][0] <= Ep < hea_bins[e][0]+bwidth:
                        g = ((2.*Jp+1.)/hea_bins[e][1])*(hea_G[e]/G)
                        break
                else:
                    print 'Error in assigning statistical weight to high energy state E={} MeV.'.format(Ep)
                    raw_input('pause')
            
            ## Get daughter state energy.
            if forward:
                nd = int(info[1])
            else:
                nd = int(info[0])
            Ed = states_d[nd-1][0]
            Jd = states_d[nd-1][1]
            ## Compute total transition energy.
            deltaE = Ed - Ep + dM
            ## Strength.
            if forward:
                B = float(info[2])
            else:
                B = ((2.*Jd+1)/(2.*Jp+1)) * float(info[2])
            
            ## Round deltaE to nearest multiple of dEwidth.
            deltaE = round(dEwidth * round(deltaE/dEwidth),10)
            ## Figure out dE bin of this transition and add to thermal_strength.
            if B > 0 and deltaE < dEmax:
                thermal_strength[:,numpy.where(dE==deltaE)[0][0]] = thermal_strength[:,numpy.where(dE==deltaE)[0][0]] + g*B
            
    ## Trim leading and trailing values of dE that contain zero strength.
    while True not in (thermal_strength[:,0] > 0):
        thermal_strength = thermal_strength[:,1:]
        dE = dE[1:]
    while True not in (thermal_strength[:,-1] > 0):
        thermal_strength = thermal_strength[:,:-1]
        dE = dE[:-1]
    
    return (thermal_strength,dE)
##### end thermal_strength #####


##### begin user_float #####
# Gets input from the user and converts to appropriate format.
# prompt is the prompt given to the user.
# notes is a parenthetical, e.g., units
# default is an optional default value.
def user_float(prompt,notes=None,default=None):
    prompt = '\n'+prompt
    if notes:
        if default:
            other = ' ({}, default {}): '.format(notes,default)
        else:
            other = ' ({}): '.format(notes)
    else:
        if default:
            other = ' (default {}): '.format(default)
        else:
            other = ': '
    
    if default:
        input = raw_input(prompt+other)
        try:
            input = float(input)
        except:
            print 'Using default.'
            input = float(default)
    else:
        while True:
            input = raw_input(prompt+other)
            try:
                input = float(input)
                break
            except:
                print 'Invalid input.'
    
    return input
##### end user_float #####