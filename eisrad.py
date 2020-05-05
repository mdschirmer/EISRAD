#!/usr/bin/env python
"""
:Summary:

:Description:
In all of the code below, A is seen as ground truth/gold standard for comparison

Selected the ones that work in extreme cases (no segmentation, i.e. ground truth vs empty image)
This is to avoid true negative issue in 'large' images, where the majority should have the label 0

filelist: man outline, aut outline

:Requires:

:TODO:
Update this meassage

:AUTHOR: MDS/MJN
:ORGANIZATION: MGH/HMS
:CONTACT: software@markus-schirmer.com
:SINCE: 2018-11-12
:VERSION: 0.1
"""
#=============================================
# Metadata
#=============================================
__author__ = 'mds'
__contact__ = 'software@markus-schirmer.com'
__copyright__ = ''
__license__ = ''
__date__ = '2019-06'
__version__ = '1.0'

#=============================================
# Import statements
#=============================================
import sys
import os
import numpy as np
import csv
import pdb

from math import pi
import matplotlib
from matplotlib import gridspec, ticker
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib.cm as cm

import nibabel as nib
import nibabel.freesurfer.mghformat as fsmgh

import skimage.measure as skm

from optparse import OptionParser

#=============================================
# Helper functions
#=============================================
def get_truepos(A,B):
    """Return true positive"""
    return np.float(np.sum(np.logical_and((B==1), (A==1)).astype(int)))


def get_trueneg(A,B):
    """Return true negative"""
    return np.float(np.sum(np.logical_and((B==0), (A==0)).astype(int)))


def get_falsepos(A,B):
    """Return false positive"""
    return np.float(np.sum(np.logical_and((B==1), (A==0)).astype(int)))


def get_falseneg(A,B):
    """Return false negative"""
    return np.float(np.sum(np.logical_and((A==1), (B==0)).astype(int)))


def get_dice(A, B):
    """Return Dice coefficient"""
    TP = get_truepos(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    return (2.*TP)/(2.*TP + FP + FN)


def get_jaccard(A, B):
    """Return Jaccard index"""
    TP = get_truepos(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    return (TP)/(TP + FP + FN)


def get_sensitivity(A,B):
    """Return sensitivity"""
    TP = get_truepos(A,B)
    FN = get_falseneg(A,B)

    return (TP/(TP + FN))


def get_specificity(A,B):
    """Return specificity"""
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)

    return (TN/(TN + FP))


def get_global_consistency_error(A,B):
    """Return global consistency error"""
    n = float(A.size)

    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    E1 = (FN*(FN+2*TP)/(TP+FN) + (FP*(FP+2*TN))/(TN+FP)) / n
    E2 = (FP*(FP+2*TP)/(TP+FP) + FN*(FN+2*TN)/(TN+FN)) / n
    
    return np.min( [E1, E2] )


def get_volumetric_similarity(A,B):
    """Return volumetric similarity"""
    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)  

    return 1.- np.abs(FN - FP)/(2.*TP + FP + FN)


def get_abcd(A,B):
    n = float(A.size)
    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)  

    a = 0.5*(TP*(TP-1) + FP*(FP-1) + TN*(TN-1) + FN*(FN-1))
    b = 0.5*((TP + FN)**2 + (TN + FP)**2 - (TP**2 + TN**2 + FP**2 + FN**2))
    c = 0.5*((TP + FP)**2 + (TN + FN)**2 - (TP**2 + TN**2 + FP**2 + FN**2))
    d = n*(n-1.)/2. - (a + b + c)

    return a,b,c,d


def get_rand_idx(A,B):
    # get a, b, c and d
    a,b,c,d = get_abcd(A,B)

    RI = (a + b)/(a + b + c + d)
    ARI = 2*(a*d - b*c)/(c**2 + b**2 + 2*a*d + (a+d)*(c+b))

    return RI, ARI


def get_probabilities(A,B):
    n = float(A.size)
    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    # p: S_g^1, S_g^2, S_t^1, S_t^2)
    p = []
    p.append((TP + FN)/n)
    p.append((TN + FN)/n)
    p.append((TP + FP)/n)
    p.append((TN + FP)/n)

    # p: (S_g^1, S_t^1), (S_g_1 S_t^2), (S_t^2, S_g^1) , (S_g^1,S_t^2)
    p.append(TP/n)
    p.append(FN/n)
    p.append(FP/n)
    p.append(TN/n)

    return p


def get_log(p):
    """Return base-2 logarithm multiplied by exponent.
    If exponent is 0, then return 0
    """
    if p==0:
        return 0.
    return p*np.log2(p)


def get_MI(A,B):
    """Return mutual information"""
    # get probabilities
    p = [get_log(ii) for ii in get_probabilities(A,B)]

    H_1 = - (p[0] + p[1])
    H_2 = - (p[2] + p[3])
    H_12 = - (np.sum(p[4:]))

    MI = H_1 + H_2 - H_12
    VOI = (H_1 + H_2 - 2*MI)/(2*np.log2(2.))

    return 2*MI/(H_1+H_2), VOI


def get_ICC(A,B):
    """Return intraclass correlation coefficient"""
    n = float(A.size)
    mean_img = (A + B)/2.

    MS_w = np.sum((A - mean_img)**2 + (B-mean_img)**2)/n
    MS_b = 2/(n-1) * np.sum((mean_img - np.mean(mean_img))**2)

    return (MS_b - MS_w)/(MS_b + MS_w)


def get_PBD(A,B):
    """Return probabilistic distance"""
    combined = np.sum(np.multiply(A,B))
    if combined==0:
        return 1

    return np.sum(np.abs(A-B))/(2.*combined)


def get_KAP(A,B):
    """Return Cohen's kappa"""
    n = float(A.size)
    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    fa = TP + TN
    fc = 1./n * ((TN + FN)*(TN+FP)+(FP+TP)*(FN+TP))

    return ((fa-fc)/(n-fc))


def get_AUC(A,B):
    """Return AUC"""
    n = float(A.size)
    TP = get_truepos(A,B)
    TN = get_trueneg(A,B)
    FP = get_falsepos(A,B)
    FN = get_falseneg(A,B)

    return 1.-0.5*(FP/(FP+TN) + FN/(FN+TP))


def directed_HD(A,B):
    """Calculate the Hausdorff distance"""
    # get coordinates
    coords_A = np.vstack(np.where(A)).transpose()
    coords_B = np.vstack(np.where(B)).transpose()
    if (len(coords_A) == 0) and (len(coords_B)==0):
        return 1.
    if (len(coords_A) == 0) or (len(coords_B)==0):
        return 1.

    #normalize by max possible distance
    max_distance = float(np.sqrt(np.sum(np.asarray(A.shape)**2)))

    # calculate all distances between points in A and B
    min_dist = []
    for ii in np.arange(coords_A.shape[0]):
        min_dist.append(np.min(np.sqrt(np.sum((coords_B-coords_A[ii,:])**2, axis=1))))
        
    return min_dist


def get_HD(A,B):
    """Return the Hausdorff distance"""
    HD_AB = np.max(directed_HD(A,B))
    HD_BA = np.max(directed_HD(B,A))

    return np.max([HD_AB, HD_BA])


def get_AVD(A,B):
    HD_AB = np.mean(directed_HD(A,B))
    HD_BA = np.mean(directed_HD(B,A))

    return np.max([HD_AB, HD_BA])


def get_ODER(A,B):
    """Return the detection and outline error rates
    see Wack et al. 2012 - Improved assessment of multiple sclerosis lesion segmentation agreement via detection and outline error estimates
    """

    # mean area of raters
    MTA = (np.sum(A) + np.sum(B))/2.

    # intersection of outlines
    intersect = np.multiply(A,B)

    # regions in A
    labels_A = skm.label(A)

    # regions in B
    labels_B = skm.label(B)

    # labels in found in A but also in B
    labels_in_A_and_B = np.unique(np.multiply(intersect, labels_A))
    labels_in_B_and_A = np.unique(np.multiply(intersect, labels_B))

    # labels unique in A and unique in B
    labels_only_in_A = np.asarray([ii for ii in np.unique(labels_A) if ii not in labels_in_A_and_B])
    labels_only_in_B = np.asarray([ii for ii in np.unique(labels_B) if ii not in labels_in_B_and_A])

    # make sure 0 is not picked up
    labels_in_A_and_B = labels_in_A_and_B[labels_in_A_and_B>0]
    labels_in_B_and_A = labels_in_B_and_A[labels_in_B_and_A>0]
    labels_only_in_A = labels_only_in_A[labels_only_in_A>0]
    labels_only_in_B = labels_only_in_B[labels_only_in_B>0]

    # calculate detection error
    # sum of areas only picked up by A plus sum of areas only picked up by B
    DE = np.sum([np.sum(labels_A==ii) for ii in labels_only_in_A]) + np.sum([np.sum(labels_B==ii) for ii in labels_only_in_B])

    # calculate outline error
    # total difference between union and intersection of the region that was outlines by both
    # = area determined by rater 1 + area determined by rater b - 2 * area determined by both
    # as union is area determined by rater 1 + area determined by rater b - area determined by both
    OE = np.sum([np.sum(labels_A==ii) for ii in labels_in_A_and_B]) + np.sum([np.sum(labels_B==ii) for ii in labels_in_B_and_A]) - 2*np.sum(intersect)

    # convert to rates and return
    return OE/MTA, DE/MTA


def get_values(A,B,measures):
    """Return all similarity metric values"""

    # initialise
    values = {}

    # run through list of implementations
    if 'Dice' in measures:
        values['Dice'] = get_dice(A,B)
    if 'Jaccard' in measures:
        values['Jaccard'] = get_jaccard(A,B)
    if 'TPR' in measures:
        values['TPR'] = get_sensitivity(A,B)
    if 'TNR' in measures:
        values['TNR'] = get_specificity(A,B)
    if 'VS' in measures:
        values['VS'] = get_volumetric_similarity(A,B)
    if '1-GCE' in measures:
        values['1-GCE'] = 1.-get_global_consistency_error(A,B)
    if ('1-VOI' in measures) or ('MI' in measures):
        NMI, VOI = get_MI(A,B)
        if '1-VOI' in measures:
            values['1-VOI'] = 1.-VOI
        if 'MI' in measures:
            values['MI'] = NMI
    if ('RI' in measures) or ('ARI' in measures):
        NRI, ARI = get_rand_idx(A,B)
        if 'RI' in measures:
            values['RI'] = NRI
        if 'ARI' in measures:
            values['ARI'] = ARI
    if 'ICC' in measures:
        values['ICC'] = get_ICC(A,B)
    if '1/(1+PBD)' in measures:
        values['1/(1+PBD)'] = 1./(1+get_PBD(A,B))
    if 'KAP' in measures:
        values['KAP'] = get_KAP(A,B)
    if 'AUC' in measures:
        values['AUC'] = get_AUC(A,B)
    if '1/(1+HD)' in measures:
        values['1/(1+HD)'] = 1./(1.+get_HD(A,B))
    if '1/(1+AVD)' in measures:
        values['1/(1+AVD)'] = 1./(1.+get_AVD(A,B))
    if ('1-OER' in measures) or ('1-DER' in measures):
        OER, DER = get_ODER(A,B)
        if '1-OER' in measures:
            values['1-OER'] = 1.-OER
        if '1-DER' in measures:
            values['1-DER'] = 1.-DER

    return values


def plot_single_evaluation(A,B, measures = ['Dice','Jaccard', 'TPR', 'TNR', '1-GCE', 'VS', 'RI', 'ARI', 'MI', '1-VOI', 'ICC','1/(1+PBD)', 'KAP', 'AUC', '1/(1+HD)', '1/(1+AVD)', 'MHD' ]):
    values_dict = get_values(A,B, measures)
    values = []
    for key in values_dict.keys():
        values.append(values_dict[key])
    measures = list(values_dict.keys())
    values = list(np.round(values,2))
    print(values)
    print(measures)

    # elements of the circle
    N = len(measures)
    x_as = [n / float(N) * 2 * pi for n in range(N)]

    # close the circle
    values += values[:1]
    x_as += x_as[:1]

    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Create polar plot
    ax = plt.subplot(111, polar=True)

    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set yticks
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)

    # Plot data
    ax.plot(x_as, values, linewidth=0, linestyle='solid', zorder=3)

    # Fill area
    ax.fill(x_as, values, 'b', alpha=0.3)

    # Set axes limits
    plt.ylim(0, 1)

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi-0.05

        if i == 1:
            ax.text(angle_rad, 1.19, measures[i], size=12, horizontalalignment='center', verticalalignment="center")
        else:
            ax.text(angle_rad, 1.14, measures[i], size=12, horizontalalignment='center', verticalalignment="center")

    # Save and show polar plot
    plt.savefig("polar_results.png",bbox_inches='tight')
    plt.show()


def plot_evaluation(values, info, measures = ['Dice','Jaccard', 'TPR', 'TNR', '1-GCE', 'VS', 'RI', 'ARI', 'MI', '1-VOI', 'ICC','1/(1+PBD)', 'KAP', 'AUC', '1/(1+HD)', '1/(1+AVD)', 'MHD' ], colourmap=None, outfile='polar_results.png'):
    """Plot radial plot for all values and measures"""
    _min = info['minimum']
    _max = info['maximum']
    if colourmap is None:
        colourmap = [[86./255.,180./255.,233./255.] for ii in range(values.shape[0])]
    else:
        # normalize colourmap values between 0 and 1
        colourmap = (colourmap-_min)/(_max-_min)
        # apply cividis, returns the RBG1 values for cividis, for dots
        colourmap = [[cm.cividis(ii)] for ii in colourmap] 

    # elements of the circle
    N = len(measures)
    # evenly space measures around circle
    x_as = [n / float(N) * 2 * pi for n in range(N)] 

    # Set color of axes
    plt.rc('axes', linewidth=0.5, edgecolor="#888888")

    # Create polar plot
    fig = plt.figure(figsize = (11,9.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[17,2,1])
    ax = plt.subplot(gs[0], polar=True)
   
    # Set position of y-labels
    ax.set_rlabel_position(0)

    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)
    ax.yaxis.grid(True, color="#888888", linestyle='solid', linewidth=0.5)

    # Set yticks
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)
    pos=ax.get_rlabel_position()
    ax.set_rlabel_position(pos+0.4*360./float(len(measures)))

    # Plot data
    for ii in np.arange(values.shape[0]):
        xx = np.asarray(x_as) + np.random.randn(len(x_as))*np.diff(x_as)[0]/15.
        data_norm = None
        if info['logplot']:
            data_norm = matplotlib.colors.LogNorm(vmin=_min, vmax=_max)
        sc = ax.scatter(xx, values[ii,:], 23, color=colourmap[ii]*len(xx), norm=data_norm, zorder=3) 

    # Fill area
    # close the circle
    median = list(np.median(values, axis=0))
    median += median[:1]
    upper = list(np.percentile(values, 75, axis=0))
    upper += upper[:1]
    lower = list(np.percentile(values, 25, axis=0))
    lower += lower[:1]
    x_as += x_as[:1]
    ax.plot(x_as, median, color=[86./255.,180./255.,233./255.], zorder=5)
    ax.fill_between(x_as, upper, lower, zorder=4, color=[86./255.,180./255.,233./255.], alpha=0.3)

    # Set number of radial axes and remove labels
    plt.xticks(x_as[:-1], [])

    # Set axes limits
    plt.ylim(0, 1)

    # Draw ytick labels to make sure they fit properly
    for i in range(N):
        angle_rad = i / float(N) * 2 * pi-0.05
        text_size = 21
        if i in {3,8}:
            ax.text(angle_rad, 1.15, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        elif i in {0}:
            ax.text(angle_rad, 1.25, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        elif i in {1,5,7}:
            ax.text(angle_rad, 1.29, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        elif i in {4}:
            ax.text(angle_rad, 1.32, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="top")
        elif i in {10}:
            ax.text(angle_rad, 1.26, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        elif i in {6}:
            ax.text(angle_rad, 1.25, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        elif i in {9}:
            ax.text(angle_rad, 1.18, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")
        else:
            ax.text(angle_rad, 1.22, measures[i]+"\n(m=%0.2f)" %median[i], size=text_size, horizontalalignment='center', verticalalignment="center")

    # colorbar location on figure
    cbaxes = plt.subplot(gs[2])

    # log scaling option
    norm = None
    if info['logplot']:
        norm = matplotlib.colors.LogNorm(vmin=_min,vmax=_max)

    img = plt.imshow(np.array([[_min,_max]]), aspect='auto', cmap="cividis", norm=norm)
    img.set_visible(False)

    # initialize colorbar
    cbar = plt.colorbar(cax = cbaxes)

    # ticks and label
    c_values = cbar.get_ticks().tolist()
    
    ticklabels = ["" for ii in c_values]
    if _min < np.min(c_values):
        c_values = [_min] + c_values
        ticklabels = ["%0.1f %s" %(np.min(c_values), info['unit'])] + ticklabels
    else:
        ticklabels[0] = "%0.1f %s" %(np.min(c_values), info['unit'])

    if _max > np.max(c_values):
        c_values = c_values + [_max]
        ticklabels = ticklabels + ["%0.1f %s" %(np.max(c_values), info['unit'])]
    else:
        ticklabels[-1] = "%0.1f %s" %(np.max(c_values), info['unit'])
    
    cbar.set_ticks(c_values)
    cbar.set_ticklabels(ticklabels)
    cbaxes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cbar.ax.set_ylabel(info["label"], labelpad=-20)
    
    # font sizes for colorbar
    cbar.ax.yaxis.label.set_size(19)
    cbar.ax.tick_params(labelsize=14)

    # Save and show polar plot 
    plt.savefig(outfile)
    if info['display']:
        plt.show()
    plt.clf()
    plt.close('all')

#=============================================
# Main method
#=============================================
def main(argv):

    ###################################
    # catch input
    ###################################

    try:
        parser = OptionParser()
        parser.add_option('-f', '--file', dest='f', help='Input FILE', metavar='FILE')
        parser.add_option('-o', '--output', dest='o', help='Output image FILE.png', metavar='FILE', default='polar_results.png')
        parser.add_option('-r', '--results', dest='r', help='Output csv file with all measures', metavar='FILE', default=None)
        parser.add_option('-m', '--min', dest='min', help='Minimum colorbar value', metavar='MIN', default=None)
        parser.add_option('-M', '--max', dest='max', help='Maximum colorbar value', metavar='MAX', default=None)
        parser.add_option('-L', '--label', dest='label', help='Label for colorbar', metavar='STRING', default='')
        parser.add_option('-l', '--log', default=False, action="store_true", help='Plot logarithmic colorbar values')
        parser.add_option('-u', '--unit', dest='unit', help='Label for colorbar', metavar='STRING', default='')
        parser.add_option('-d', '--display', default=True, action='store_false', help='Display the output before saving as png')
        parser.add_option('-v', '--verbose', default=False, action="store_true", help='verbose output')
        parser.add_option('-b', '--binarize', dest='binarize', default=False, action="store_true", help='binarize input images')
        (options, args) = parser.parse_args()
    except Exception as e:
        print(e)
        print('Call help with the -h option')
        sys.exit(2)

    if options.f is None:
        print('Need at least an input file. Call -h.')
        sys.exit()

    seg_file = options.f
    outfile = options.o

    if (not os.path.isdir(os.path.dirname(outfile))) and (not os.path.dirname(outfile)==''):
        print('Output folder %s not found.' %outfile)
        sys.exit()

    if not os.path.isfile(seg_file):
        print('Input file %s not found.' %nii_file)
        sys.exit()

    if options.min is not None:
        try:
            min_val = float(options.min)
        except:
            print('Minimum value for colormap is not a number (%s).' %options.min)
            sys.exit()
    else:
        min_val = None

    if options.max is not None:
        try:
            max_val = float(options.max)
        except:
            print('Maximum value for colormap is not a number (%s).' %options.max)
            sys.exit()
    else:
        max_val = None

    colorbar_label = options.label
    logplot = options.log
    unit = options.unit
    display = options.display

    # load file
    with open(seg_file,'r') as fid:
        reader = csv.reader(fid)
        file_list = [[row[0],row[1]] for row in reader]

    # write necessary information into dictionary
    info = {'minimum': min_val, 'maximum': max_val, 'label': colorbar_label, 'logplot': logplot, 'unit': unit, 'display': display}

    # define measures to use                
    measures = ['Dice', 'Jaccard', 'TPR', 'VS', 'MI', 'ARI', 'ICC', '1/(1+PBD)', 'KAP', '1-OER', '1-DER']

    # initialize outputs
    data = []
    colourmap = []

    # loop through each line in the file for comparison
    for isub,sub in enumerate(file_list):
        aut_file = sub[0]
        man_file = sub[1]

        # verbose to stdout
        if options.verbose:
            print ("Processing subject #%s" %isub)
            print ("Automated file: %s" %sub[0])
            print ("Manual file: %s" %sub[1])

        # if either file is not found, skip
        if not (os.path.isfile(man_file) and os.path.isfile(aut_file)):
            print('no outline %s' %sub)
            continue

        # load image files for each subject

        ## rater 1
        A = nib.load(man_file).get_data()
        if options.binarize:
            A = (A>0).astype(int)
        ## rater 2
        nii = nib.load(aut_file)
        B = nii.get_data()
        if options.binarize:
            B = (B>0).astype(int)

        # keep track of volume for colorbar mapping
        vol_cc = float(np.sum(B)*np.prod(nii.get_header().get_zooms()))/1000
        colourmap.append(vol_cc)

        if options.verbose:
            print ("Volume is %s cc" %vol_cc)

        # compare raters and make them ready for plotting
        values_dict = get_values(A,B, measures)
        values = []
        measures = []
        for key in sorted(list(values_dict.keys())):
            values.append(values_dict[key])
            measures.append(key)
        values = list(np.round(values,2))
        data.append(values)

    if options.r is not None:
        with open(options.r, 'w') as fid:
            writer = csv.writer(fid)
            writer.writerow(measures)
            writer.writerows(data)

    # update if minimum and maximum were not passed as an argument
    if info['minimum'] is None:
        info['minimum'] = round(np.min(colourmap),2)
    if info['maximum'] is None:
        info['maximum'] = round(np.max(colourmap),2)

    # create plot
    plot_evaluation(np.asarray(data), info, measures, np.asarray(colourmap), outfile=outfile)


if __name__ == "__main__":
    main(sys.argv)
