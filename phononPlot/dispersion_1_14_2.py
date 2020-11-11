#!/usr/bin/env python
#coding=utf-8
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import OrderedDict
from pango import Weight
from matplotlib.pyplot import subplot
from progressbar import ProgressBar, Percentage, Bar, ETA

class PhononDispersion():
    def __init__(self, path):
        self.path=path
    
    def read(self):
        infile=open(self.path)
        
        band=OrderedDict()
        
        # read data
        band['nqpoint']=int(infile.readline().split()[1])
        npath=int(infile.readline().split()[1])
        infile.readline()
        for i in xrange(0, npath):
            infile.readline()
        # reciprocal lattice
        reciprocal_lattice=np.zeros((3,3))
        infile.readline()
        for i in xrange(0, 3):
            reciprocal_lattice[i]=np.array([float(s0) for s0 in infile.readline().split('[')[1].split(']')[0].split(',')])
        band['reciprocal_lattice']=reciprocal_lattice
        
        band['natom']=int(infile.readline().split()[1])
        
        # lattice
        lattice=np.zeros((3,3))
        infile.readline()
        for i in xrange(0, 3):
            lattice[i]=np.array([float(s0) for s0 in infile.readline().split('[')[1].split(']')[0].split(',')])
        band['lattice']=lattice
        
        distances=np.zeros((band['nqpoint']))    
        phonons=np.zeros((band['nqpoint'], 3*band['natom']))
        hsp={'symbol':[], 'distance':[]} # high symmetry points
        eigenvectors=np.zeros((band['nqpoint'],3*band['natom'],band['natom'],3), dtype=complex)
        
        qcounter=0
        bcounter=0
        # skip some data
        string=infile.readline()
        while(not string.startswith('phonon')):
            string=infile.readline()    
        # read phonon
        string=infile.readline()
        while(string):    
            # read q point
            if string.startswith('- q-position:'): # enter a qpoint
                distance=float(infile.readline().split()[1])
                distances[qcounter]=distance
                string=infile.readline().split()
                if string[0].startswith('label:'):
                    label=string[1].split('\'')[1]
                    if label == '\Gamma' or label == 'Gamma':
                        label='$\Gamma$'
                    hsp['symbol'].append(label)
                    hsp['distance'].append(distance)
                    string=infile.readline().split()
                # band
                if string[0].startswith('band:'):
                    string=infile.readline()
                    while(string):
                        string=string.split()
                        if string[0].startswith('frequency:'):
                            frequency=float(string[1])
                            phonons[qcounter][bcounter]=frequency
                            
                            # eigenvector
                            string=infile.readline()
                            while (not string.split()[0].startswith('eigenvector:')):
                                string=infile.readline()
                            for atom in xrange(0, band['natom']):
                                infile.readline()
                                for ev in xrange(0, 3):
                                    tmp=[float(s0) for s0 in infile.readline().split('[')[1].split(']')[0].split(',')]
                                    eigenvectors[qcounter][bcounter][atom][ev]=complex(tmp[0], tmp[1])
                            bcounter += 1
                            if bcounter == 3*band['natom']:
                                qcounter += 1
                                bcounter=0
                                break
                        string=infile.readline()    
            string=infile.readline()
            
        band['hsp']=hsp
        band['distances']=distances
        band['phonons']=np.transpose(phonons) # [3*natom, nqpoint]
        band['eigenvectors']=eigenvectors # [nqpoint, nband, 3*natom, 3]
        
        infile.close()
        return band

    def plot(self, subplot):
        band=self.read()
        distances=band['distances']
        phonons=band['phonons']
        hsp=band['hsp']
        
        for i in xrange(0, phonons.shape[0]):
            subplot.plot(distances, phonons[i], '-', lw=2, c='k')
            
        hspcoordinate=hsp['distance']
        for i in xrange(0, len(hspcoordinate)):
            subplot.axvline(x=hspcoordinate[i], c='k')
        symbol=hsp['symbol']
        subplot.set_xlim(hspcoordinate[0],hspcoordinate[-1])
        subplot.set_xticks(hspcoordinate)
        subplot.set_xticklabels(symbol, fontsize=16)
        subplot.tick_params(labelsize=14)
        
#    def plotPDOS(self, subplot, atoms):
#        """
#        Arguments:
#            atoms: list of atoms. e.g.: 1, 2, 3 4 5
#        """
#        band=self.read()
#        distances=band['distances']
#        phonons=band['phonons']
#        eigenvectors=band['eigenvectors']
#        
#        tmp=atoms.split(',')
#        for i in xrange(0, len(timp)):
            
            
class PDOS():            
    def __init__(self, path):
        self.path=path
    
    def read(self):
        infile=open(self.path)

        elements=OrderedDict()        
        pdos=OrderedDict()
        
        string=infile.readline() # mesh
        pdos['nqpoint']=int(infile.readline().split()[1])
        # reciprocal lattice
        infile.readline()
        for i in xrange(0, 3):
            infile.readline()
        
        pdos['natom']=int(infile.readline().split()[1])
        
        # lattice
        infile.readline()
        for i in xrange(0, 3):
            infile.readline()
            
        # points
        string=infile.readline()
        if string.startswith('points:'):
            string=infile.readline()
            while(string):
                if string.split()[0].startswith('phonon:'):
                    break
                elif string.split()[1].startswith('symbol:'):
                    string=string.split()
                    if string[2] in elements.keys():
                        elements[string[2]]=np.hstack((elements[string[2]], int(string[4]))).tolist()
                    else:
                        elements[string[2]]=[int(string[4])]
                
                string=infile.readline()
        pdos['elements']=elements
        weights=np.zeros((pdos['nqpoint']))
        frequencies=np.zeros((pdos['nqpoint'],3*pdos['natom']))
        eigenvectors=np.zeros((pdos['nqpoint'],3*pdos['natom'],pdos['natom'],3), dtype=complex)
        
        qcounter=0
        bcounter=0
        while(string):
            # read q point
            if string.startswith('- q-position:'): # enter a qpoint
                infile.readline() # distance
                weights[qcounter]=float(infile.readline().split()[1])
                
                string=infile.readline()
                while(string):
                    string=string.split()
                    if string[0].startswith('frequency:'):
                        frequencies[qcounter][bcounter]=float(string[1])
                        
                        string=infile.readline().split()
                        while(not string[0].startswith('eigenvector:')):
                            string=infile.readline().split()
                            
                        for atom in xrange(0, pdos['natom']):
                            infile.readline()
                            for k in xrange(0, 3):
                                tmp=np.array([float(s0) for s0 in infile.readline().split("[")[1].split("]")[0].split(",")])
                                eigenvectors[qcounter][bcounter][atom][k]=complex(tmp[0], tmp[1])
                        
                        bcounter += 1
                        if bcounter == 3*pdos['natom']:
                            qcounter += 1
                            bcounter=0
                            break
                            
                    string=infile.readline()
                    
            string=infile.readline()
        
        # phdos
        frequencies, phdos=self.calculate(pdos['nqpoint'], pdos['natom'], weights, frequencies, eigenvectors)
        
        pdos['frequencies']=frequencies
        pdos['phdos']=phdos
        
        infile.close()
        return pdos

    def calculate(self, nqpoint, natom, weights, frequencies, eigenvectors):
        fmax=np.max(frequencies)
        fmin=np.min(frequencies)
        sigma=(fmax-fmin)/100
        
        weights=weights/nqpoint
        eigenvectors2=np.zeros((nqpoint, 3*natom, natom))
        
        for i in xrange(0, nqpoint):
            for j in xrange(0, 3*natom):
                for k in xrange(0, natom):
                    eigenvectors2[i][j][k]=np.abs(eigenvectors[i][j][k][0])**2 + np.abs(eigenvectors[i][j][k][1])**2 + np.abs(eigenvectors[i][j][k][2])**2
        f_delta=(fmax-fmin)/200
        freq=np.arange(fmin-sigma*10, fmax+sigma*10+f_delta*0.1, f_delta)
        
        # phdos
        phdos=[]
        for l in xrange(0, freq.shape[0]):
            amplitude=1.0/np.sqrt(2*np.pi)/sigma*np.exp(-(frequencies-freq[l])**2/2.0/sigma**2)
            t=[]
            for i in xrange(0, natom):
                tmp=np.dot(weights, eigenvectors2[:,:,i]*amplitude).sum()
                t.append(tmp)
        
            if phdos == []:
                phdos=np.array(t)
            else:
                phdos=np.vstack((phdos,t))
            
        phdos=np.transpose(phdos)
        frequencies=freq
        
        return frequencies, phdos
    
    def plot(self, subplot,linestyle, color, ordered_elements=None):
        pdos=self.read()
        elements=pdos['elements']
        freq=pdos['frequencies']
        phdos=pdos['phdos']
        
        total_dos=np.zeros((phdos.shape[1]))
        ecounter=0 # counter of element

        if ordered_elements is None:
            symbols=elements.keys()
        else:
            symbols=ordered_elements
        for symbol in symbols:
            dtmp=np.zeros((phdos.shape[1]))
            for atom in elements[symbol]:
                dtmp += phdos[atom-1]
            total_dos += dtmp
            subplot.plot(dtmp, freq, lw=2, linestyle=linestyle[ecounter], c=color[ecounter], label=symbol)
            ecounter += 1
        #subplot.plot(total_dos, freq, lw=2, linestyle=linestyle[ecounter], c=color[ecounter], label='total')
        

class Gruneisen():
    def __init__(self, path, natom):
        self.path=path
        self.natom=natom
    
    def read(self):
        infile=open(self.path)
        
        npath=int(os.popen("grep nqpoint " + self.path + " | wc -l").readline())
        nqpoint=int(os.popen("grep nqpoint " + self.path).readline().split()[2])
        
        nband=self.natom*3
        
        distances=np.zeros((npath,nqpoint))
        frequencies=np.zeros((npath,nqpoint,nband))
        gruneisens=np.zeros((npath,nqpoint,nband))
        
        path=0 # path counter
        qpoint=0 # qpoint counter
        
        # comment lines
        string=infile.readline()
        while (string):
            string=infile.readline()
            
            if string.startswith("  - q-position:"):
                q=[True if np.abs(float(s0)-0.0) < 1e-6 else False for s0 in string.split('[')[1].split(']')[0].split(',')]
                # distance
                if path == 0:
                    distances[path][qpoint]=float((infile.readline().split())[1])
                else:
                    distances[path][qpoint]=float((infile.readline().split())[1])+distances[path-1][-1]
                # read band
                infile.readline() # skip 'band:'
                for i in range(0, nband):
                    string=infile.readline()
                    if (string != "/n"):
                        number=int((string.split())[2]) # band counter
                        gruneisens[path][qpoint][number-1]=float((infile.readline().split())[1])
                        frequencies[path][qpoint][number-1] = float((infile.readline().split())[1])
                        if not(False in q) and (number <= 3):
                            gruneisens[path][qpoint][number-1]=0.0

                if (qpoint == nqpoint-1):
                    path += 1
                    qpoint = 0
                elif (qpoint < nqpoint-1):
                    qpoint += 1
 
        gruneisen={'distances':distances, 'frequencies':frequencies, 'gruneisens':gruneisens}
        return gruneisen

    def plot(self, subplot):
        gruneisen=self.read()
        distances=gruneisen['distances']
        frequencies=gruneisen['frequencies']
        gruneisens=gruneisen['gruneisens']
        
        npath=frequencies.shape[0]
        nqpoint=frequencies.shape[1]
        nband=frequencies.shape[2]
        for i in xrange(0, npath):
            for j in xrange(0, nqpoint):
                for k in xrange(0, nband):
                    if gruneisens[i,j,k] < 0:
                        subplot.scatter(distances[i,j],frequencies[i,j,k],s=-gruneisens[i,j,k],color='r', lw=0.2, alpha=1, edgecolors='k')
                    else:
                        subplot.scatter(distances[i,j],frequencies[i,j,k],s=gruneisens[i,j,k],color='b', lw=0.2, alpha=1, edgecolors='k')
        
        subplot.set_xlim(np.min(distances), np.max(distances))

        
class GruneisenByHand():
    def __init__(self, path):
        self.path=path
        
    def read(self):
        """
        Note that the 'minus', 'orig' and 'plus' directories need to be exist under the give path.
        """
        def getVolume(lattice):
            return np.linalg.det(lattice)
        
        band_m=PhononDispersion(self.path+'/minus/band.yaml').read()
        band_o=PhononDispersion(self.path+'/orig/band.yaml').read()
        band_p=PhononDispersion(self.path+'/plus/band.yaml').read()
        
        distances_m=band_m['distances']
        phonons_m=band_m['phonons']
        v_m=getVolume(band_m['lattice'])
        
        distances_o=band_o['distances']
        phonons_o=band_o['phonons'] # [3*natom, nqpoint]
        v_o=getVolume(band_o['lattice'])
        
        distances_p=band_p['distances']
        phonons_p=band_p['phonons']
        v_p=getVolume(band_p['lattice'])
        
        gruneise=OrderedDict()
        gruneise['distances']=distances_o
        gruneise['phonons']=phonons_o
        
        gamma=np.zeros(phonons_o.shape)
        for i in xrange(0, phonons_o.shape[0]): # band
            for j in xrange(0, phonons_o.shape[1]): # qpoint
                if phonons_p[i][j] < 1e-3:
                    gamma[i][j]=0
                else:
                    gamma[i][j]=-(v_o/phonons_o[i][j])*((phonons_p[i][j]-phonons_m[i][j])/(v_p-v_m))
        gruneise['gamma']=gamma
        
        return gruneise
    
    def plotScatteredGruneisen(self, subplot):
        """
        Arguments:
            isInvers (default=False): whether to invert the color of scatters.
        """
        gruneise=self.read()
        
        distances=gruneise['distances']
        phonons=gruneise['phonons']
        gamma=gruneise['gamma'] # Gruneisen parameters
        
        # plot Gruneisen
        nq=0 # counter of kpoint
        pbar=ProgressBar(widgets=['plotting scattered Gruneisen:', Percentage(), ' ', Bar(), 
                                  ' ', ETA()], maxval=gamma.shape[0]*gamma.shape[1]).start()
        for i in xrange(0, gamma.shape[0]): # band
            for j in xrange(0, gamma.shape[1]): # qpoint
                # g+: blue; g-: red
                if gamma[i][j] < 0:
                    subplot.scatter(distances[j], phonons[i][j], s=-gamma[i][j], c='r')
                else:
                    subplot.scatter(distances[j], phonons[i][j], s=gamma[i][j], c='b')
                nq += 1
                pbar.update(nq)
        pbar.finish()
        
            
    def plotGradientGruneisen(self, subplot, **kwargs):
        """
        Gruneisen parameters: (default)
            g+: blue; g-: red.
        Note that the corresponding gruneisen's value to two endpoints (red and blue) are same. e.g. [-10, 10]
        
        Arguments:        
            kwargs:
                g_abs_max: absolute maximum of Gruneisen parameters.
        """
        
        gruneise=self.read()
        
        distances=gruneise['distances']
        phonons=gruneise['phonons']
        gamma=gruneise['gamma'] # Gruneisen parameters
        gamma_min=np.min(gamma)
        gamma_max=np.max(gamma)
        print 'minimum of Gruneisen: %.4f' %gamma_min
        print 'maximum of Gruneisen: %.4f' %gamma_max 
        
        g_abs_max=None
        if 'g_abs_max' in kwargs:
            g_abs_max=kwargs['g_abs_max']
        else:
            g_abs_max=np.max([np.abs(gamma_min), np.abs(gamma_max)])
            
        # plot Gruneisen
        nq=0 # counter of kpoint
        pbar=ProgressBar(widgets=['plotting gradient Gruneisen:', Percentage(), ' ', Bar(), 
                                  ' ', ETA()], maxval=gamma.shape[0]*gamma.shape[1]).start()
        for i in xrange(0, gamma.shape[0]): # band
            for j in xrange(0, gamma.shape[1]-1): # qpoint
                # g+: blue; g-: red
                if gamma[i][j] < 0:
                    subplot.plot(distances[j:j+2], phonons[i][j:j+2], c=[-gamma[i][j]/g_abs_max, 0, 0])
                else:
                    subplot.plot(distances[j:j+2], phonons[i][j:j+2], c=[0, 0, gamma[i][j]/g_abs_max])
                nq += 1
                pbar.update(nq)
        pbar.finish()
        
        
# ------------------- test --------------------
#g=GruneisenByHand('/home/fu/workspace/Ferroelectric/exp/KTaO3/c/gruneisen')
#g.read()
        
