########## Computes neutrino production and energy loss rates from lepton capture and decay as a function of temperature and rhoYe. ##########
########## Rates are in units of (neutrinos, MeV)/s baryon. ##########
import numpy,os,sys,weaklib

##### Get user inputs. #####
## Get format for overlap files.
format = weaklib.choose('overlap file format',['bigstick','oxbash'])

lpt = ''
if format == 'oxbash':
    ## Get path to lpt files.
    default = './'
    lptpath = raw_input('Path to lpt file(s) (default {}): '.format(default))
    if lptpath == '':
        lptpath = default

    ## Get parent .lpt.
    print '\nParent nucleus .lpt file: '
    lpt = weaklib.get_lpt(path=lptpath)
    if not lpt:
        print 'No .lpt file found.'
        sys.exit()

## Get temperature-density grid.  Temperature will be in MeV, rhoYe in g/cm^3.
if weaklib.choose('temperature-density grid',['FFN','input temperatures and densities']) == 'FFN':
    temp = weaklib.FFNtemp * 10**9 * weaklib.kB
    rhoYe = weaklib.FFNrhoYe
else:
    ## If to be input, get temperature.
    temp_unit = weaklib.choose('unit of temperature',['MeV','T9'])
    default = '0.01'
    temp = raw_input('\nTemperatures ({}, temp1,temp2,..., default {}): '.format(temp_unit,default))
    temp = numpy.fromstring(temp,sep=',')
    if len(temp) == 0:
        print 'Using default.'
        temp = numpy.fromstring(default,sep=',')
    if temp_unit == 'T9':
        temp = temp * 10**9 * weaklib.kB
    ## If to be input, get rhoYe.
    default = '1'
    rhoYe = raw_input('\nDensity log(rho*Ye) (g/cm^3, log(rhoYe1),log(rhoYe2),... default {}): '.format(default))
    rhoYe = numpy.fromstring(rhoYe,sep=',')
    if len(rhoYe) == 0:
        print 'Using default.'
        rhoYe = numpy.fromstring(default,sep=',')
    rhoYe = 10.**rhoYe

## Get parent A,Z.
if format == 'oxbash' and lpt.startswith('A,Z='):
    AZ = lpt.split()[0].split('=')[1]
    A,Z = numpy.fromstring(AZ,sep=',',dtype=int)
else:
    while True:
        AZ = raw_input('Nucleus A,Z: ')
        try:
            A,Z = numpy.fromstring(AZ,sep=',',dtype=int)
            break
        except:
            print 'Invalid input.'
Tz = (2.*Z - A)/2.

## Get path to overlap files.
while True:
    if format == 'oxbash':
        default = './overlaps/A,Z={}/'.format(AZ)
    elif format == 'bigstick':
        default = os.path.expanduser('~/Documents/Johnson data/allmytransitions-master/Beta_Results_Final/gtr_expt_rev/')
    overlappath = raw_input('\nPath to overlap files (default {}): '.format(default))
    if os.path.exists(overlappath):
        break
    elif overlappath == '':
        print 'Using default.'
        overlappath = default
        break
    else:
        print 'Path not found.'

#####   THIS HAS NOT YET BEEN IMPLEMENTED.   #####
## Determine whether to provide rates for individual states.
individual = False
#individual = weaklib.choose('output behavior',['thermally sum initial states','individual states'])
#if individual == 'individual states':
#    individual = True
#else:
#    individual = False
####################################################

## Get initial state energies.
if individual:
    default = '0,5'
    Erange = raw_input('\nMin,max initial state energy (MeV, default {}): '.format(default))
    Erange = numpy.fromstring(Erange,sep=',')
    if len(Erange) != 2:
        print 'Using default.'
        Erange = numpy.fromstring(default,sep=',')
else:
    default = '15,20'
    cutoff = numpy.fromstring(raw_input('\nLower,upper cutoffs for modified Brink hypothesis (MeV, default {}): '.format(default)),sep=',')
    if len(cutoff) != 2:
        print 'Using default.'
        cutoff = numpy.fromstring(default,sep=',')
    hea_width = weaklib.user_float('Averaged high energy state bin width',notes='MeV',default=1.)
    
## Get resolution for transition energy binning in thermal strength.
dEwidth = weaklib.user_float('Thermal strength transition energy bin width',notes='MeV',default=0.01)

##### User inputs gotten. #####


##### main program #####
## Make output directory if it doesn't exist.
if not os.path.exists('./results/A,Z={}'.format(AZ)):
    os.makedirs('./results/A,Z={}'.format(AZ))

## Channels included in calculation.
channels = ['gt-','gt+']

## Get thermal strength for each channel.
## strength[i] = ([thermal strength i],[Q i])
thermal_strength = []
for channel in channels:
    if format == 'oxbash':
        thermal_strength.append(weaklib.thermal_strength(AZ,channel,temp,overlappath,format,lptpath=lptpath,cutoff1=cutoff[0],cutoff2=cutoff[1],bwidth=hea_width,dEmax=50.,dEwidth=dEwidth))
    elif format == 'bigstick':
        thermal_strength.append(weaklib.thermal_strength(AZ,channel,temp,overlappath,format,cutoff1=cutoff[0],cutoff2=cutoff[1],bwidth=hea_width,dEmax=50.,dEwidth=dEwidth))
    ## Convert strengths to 1/ft.
    if thermal_strength[-1]:
        for t in range(len(temp)):
            thermal_strength[-1][0][t] = thermal_strength[-1][0][t]/weaklib.B_to_ft

## Compute that rate.
for rY in range(len(rhoYe)):
    print 'Computing rates at density {} of {}.'.format(rY+1,len(rhoYe))
    
    ## Compute electron chemical potential.
    mue = []
    for t in range(len(temp)):
        mue.append(weaklib.electron_chemical_potential(rhoYe[rY],temp[t]))
    mue = numpy.array(mue)
    
    ecap = numpy.zeros((len(temp),2))    ## Electron capture neutrino production and energy loss rates.
    pdec = numpy.zeros((len(temp),2))    ## Positron decay neutrino production and energy loss rates.
    pcap = numpy.zeros((len(temp),2))    ## Positron capture neutrino production and energy loss rates.
    edec = numpy.zeros((len(temp),2))    ## Electron decay neutrino production and energy loss rates.
    
    for channel in channels:
        ## If there's no daughter for this channel, there will be no strength.
        if not thermal_strength[channels.index(channel)]:
            continue
        
        strength,Q = thermal_strength[channels.index(channel)]
        
        ## Compute captures.
        if channel == 'gt-':
            lepton = 'electron'
        elif channel == 'gt+':
            lepton = 'positron'
    
        cap = weaklib.rates_cc(A,Z,strength,temp,Q,mue,lepton,'capture')
        ## Make capture rates per baryon.
        cap = cap/A
        
        if lepton == 'electron':
            ecap = ecap + cap
        elif lepton == 'positron':
            pcap = pcap + cap
        
        ## Compute decays.
        if channel == 'gt-':
            lepton = 'positron'
        elif channel == 'gt+':
            lepton = 'electron'
        
        dec = weaklib.rates_cc(A,Z,strength,temp,Q,mue,lepton,'decay')
        ## Make decay rates per baryon.
        dec = dec/A
        
        if lepton == 'electron':
            edec = edec + dec
        elif lepton == 'positron':
            pdec = pdec + dec
            
    ## Save results to disk here if computing total rate.
    if not individual:
        
        f_out_path = './results/A,Z={}/beta rates A,Z={}.txt'.format(AZ,AZ)
        
        if not os.path.exists(f_out_path):
            f = open(f_out_path,'w+')
            f.write('rhoYe is density times electron fraction [g/cm^3], T is temperature [MeV], mu_e is electron chemical potential [MeV]\n')
            f.write('e-(+) cap is electron (positron) capture, e-(+) dec is electron (positron) decay\n')
            f.write('For each process, first column is neutrino production [neutrinos/baryon second] and second column is energy loss [MeV/baryon second].\n')
            f.write('{:<10}{:<7}{:<8}{:<25}{:<25}{:<25}{:<25}\n'.format('rhoYe','T','mu_e','e- cap','e+ dec','e+ cap','e- dec'))
        else:
            f = open(f_out_path,'a+')
            
        for t in range(len(temp)):
            f.write('{:<10.2E}{:<7.3f}{:<8.3f}{:<12.4E}{:<13.4E}{:<12.4E}{:<13.4E}{:<12.4E}{:<13.4E}{:<12.4E}{:<13.4E}\n'.format(rhoYe[rY],temp[t],mue[t],ecap[t][0],ecap[t][1],pdec[t][0],pdec[t][1],pcap[t][0],pcap[t][1],edec[t][0],edec[t][1]))
            
        ### Eliminate duplicate entries.
        f.seek(0)
        ## Length of file header.
        head_length = 4
        for _ in range(head_length):
            f.readline()
        lines = []
        for line in f:
            lines.append(line)
        lines = list(set(lines))
        lines.sort(key=lambda x: (float(x.split()[0]),float(x.split()[1])))
        f.seek(0)
        for _ in range(head_length):
            f.readline()
        f.truncate()
        for line in lines:
            f.write(line)    
        
        f.close()