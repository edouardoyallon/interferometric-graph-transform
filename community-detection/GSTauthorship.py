# 2019/05/22~23
# Fernando Gama, fgama@seas.upenn.edu

# Graph scattering transform.

# Compute the classification accuracy of the different representations on the
# authorship attribution problem
# Representations considered:
#   GFT: unstable graph-dependent representation
#   Diffusion scattering: comparison with other works
#   Monic cubic polynomial wavelet: Hammond et al wavelets
#   Tight Hann wavelets: Shuman et al wavelets
# The idea is just to show that the graph scattering transform still achieves
# reasonable classification accuracy when compared to other methods like the
# GFT or the data itself.

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pickle
import datetime

from sklearn.svm import LinearSVC

#\\\ Own libraries:
import Modules.graphScattering as GST
import Utils.graphTools as graphTools
import Utils.dataTools

#\\\ Own separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

from utils_compute_interferometric import *

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################

authorName = 'austen'
# Possible authors: (just use the names in ' ')
# jacob 'abbott', robert louis 'stevenson', louisa may 'alcott',
# horatio 'alger', james 'allen', jane 'austen', emily 'bronte', james 'cooper',
# charles 'dickens', hamlin 'garland', nathaniel 'hawthorne', henry 'james',
# herman 'melville', 'page', herny 'thoreau', mark 'twain',
# arthur conan 'doyle', washington 'irving', edgar allan 'poe',
# sarah orne 'jewett', edith 'wharton'

thisFilename = 'GSTauthorship' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location where to save
    # anything that might need to be saved out of the run
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all the
    # results from each run
dataPath = os.path.join('datasets','authorshipData','authorshipData.mat')

#\\\ Create .txt to store the values of the parameters of the setting for easier
#    reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # Append date and time of the run to the directory, to avoid several runs of
    # overwritting each other.
saveDir = saveDir + today + authorName
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########

nClasses = 2 # Either authorName or not
beginRatioTrainSim = 0.2
endRatioTrainSim = 0.9
nSimPoints =  10
ratioTrain = np.linspace(beginRatioTrainSim, endRatioTrainSim, nSimPoints)
    # Ratio of training samples
ratioValid = 0.1 # Ratio of validation samples (out of the total training
# samples)
# Final split is:
#   nValidation = round(ratioValid * ratioTrain * nTotal)
#   nTrain = round((1 - ratioValid) * ratioTrain * nTotal)
#   nTest = nTotal - nTrain - nValidation

nDataSplits =  20 # Number of data realizations
# Obs.: The built graph depends on the split between training, validation and
# testing. Therefore, we will run several of these splits and average across
# them, to compute the variation with respect to the split.

# Every training excerpt has a WAN associated to it. We combine all these WANs
# into a single graph to use as the supporting graph for all samples. This
# combination happens under some extra options:
graphNormalizationType = 'rows' # or 'cols' - Makes all rows add up to 1.
keepIsolatedNodes = False # If True keeps isolated nodes
forceUndirected = True # If True forces the graph to be undirected (symmetrizes)
forceConnected = True # If True keeps the largest connected component

#\\\ Save values:
writeVarValues(varsFile,
               {'authorName': authorName,
                'nClasses': nClasses,
                'beginRatioTrainSim': beginRatioTrainSim,
                'endRatioTrainSim': endRatioTrainSim,
                'nSimPoints': nSimPoints,
                'ratioValid': ratioValid,
                'nDataSplits': nDataSplits,
                'graphNormalizationType': graphNormalizationType,
                'keepIsolatedNodes': keepIsolatedNodes,
                'forceUndirected': forceUndirected,
                'forceConnected': forceConnected})
    
#################
# ARCHITECTURES #
#################

# Select which wavelets to use
doDiffusion = True # F. Gama, A. Ribeiro, and J. Bruna, "Diffusion scattering
    # transforms on graphs,” in Int. Conf. Learning Representations 2019.
    # New Orleans, LA: Assoc. Comput. Linguistics, 6-9 May 2019.
doMonicCubic = True # Eq. (65) in D. K. Hammond, P. Vandergheynst, and
    # R. Gribonval, "Wavelets on graphs via spectral graph theory," Appl.
    # Comput. Harmonic Anal., vol. 30, no. 2, pp. 129–150, March 2011.
doTightHann = True # Example 2, p. 4226 in D. I. Shuman, C. Wiesmeyr,
    # N. Holighaus, and P. Vandergheynst, "Spectrum-adapted tight graph wavelet
    # and vertex-frequency frames,” IEEE Trans. Signal Process., vol. 63,
    # no. 16, pp. 4223–4235, Aug. 2015.
doGFT = True # Compare against the GFT which is a (unstable) representation that
    # also depends on the graph
normalizeGSOforGFT = True # The GSO for the GFT is the Laplacian (if possible),
    # if not, it becomes the adjacency matrix. In either case, setting True
    # to this flag, gets the GSO normalized. Since we're usually comparing
    # against normalized matrix descriptions in the other cases, we give this
    # options.
doData = False # Do classification straight into the data, without any other
    # representation

doInterferometric=True

numScales = 6 # Number of scales J (the first element might be the "low-pass"
    # wavelet) so we would get J-1 "wavelet scales" and 1 (the first one, j=0)
    # "low-pass" wavelet
numLayers = 3 # Number of layers L (0, ..., L-1) with l=0 being just Ux
nFeatures = np.sum(numScales ** np.arange(0, numLayers, dtype=np.float)) 
nFeatures = np.int(nFeatures) # Number of features
fullGFT = False # if True use all the GFT coefficients
nGFTcoeff = nFeatures # number of GFT coefficients to use (if not fullGFT)
GFTfilterType = 'low' # 'low', 'band' or 'high' pass (lowest frequencies,
    # middle frequencies, or high frequencies)

#\\\ Save values:
writeVarValues(varsFile, {'numScales': numScales,
                          'numLayers': numLayers,
                          'nFeatures': nFeatures,
                          'fullGFT': fullGFT,
                          'nGFTcoeff': nGFTcoeff,
                          'GFTfilterType': GFTfilterType})

modelList = [] # List to store the list of models chosen

# Obs.: These are the names that will appear in the legend of the figure
if doDiffusion:
    diffusionName = 'Diffusion'
    modelList.append(diffusionName)
if doMonicCubic:
    monicCubicName = 'Monic Cubic'
    modelList.append(monicCubicName)
if doTightHann:
    tightHannName = 'Tight Hann'
    modelList.append(tightHannName)
if doGFT:
    GFTname = 'GFT'
if doData:
    DataName = 'Data'

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
figSize = 5 # Overall size of the figure that contains the plot
lineWidth = 2 # Width of the plot lines
markerShape = 'o' # Shape of the markers
markerSize = 3 # Size of the markers

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'figSize': figSize,
                'lineWidth': lineWidth,
                'markerShape': markerShape,
                'markerSize': markerSize,
                'saveDir': saveDir})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Save variables during evaluation.

accGST = {} # Classification accuracy
    # This is a dictionary where each key corresponds to one of the models,
    # and each element in the dictionary is a list of lists storing the
    # mean representation error (averaged across nTest) for each nTrain value,
    # for each random data split (at that fixed nTrain)
for thisModel in modelList: # First list is for each graph realization
    accGST[thisModel] = [None] * nSimPoints
# The GFT representation error is on a different variable since it cannot be
# computed in the same for loop as the rest of the model (that would require
# creating a "scattering GFT" which makes no sense)
if doGFT:
    accGFT = [None] * nSimPoints
# Same for classification straight using data
if doData:
    accData = [None] * nSimPoints

if doInterferometric:
    accInterferometric = [None] * nSimPoints


#%%##################################################################
#                                                                   #
#                    NTRAIN VALUE                                   #
#                                                                   #
#####################################################################

# For each value of nTrain (number of selected training samples)

for itN in range(nSimPoints):
    
    thisRatioTrain = ratioTrain[itN]

    if doPrint:
        print("Ratio of training samples: %.4f (no. %d)"%(thisRatioTrain,itN+1))

    # The accBest variable, for each model, has a list with a total number of
    # elements equal to the number of ratio trains that we will consider.
    # Now, for each ratio, we have multiple data splits so we want, for each
    # ratio train, to create a list to hold each of those values 
    for thisModel in modelList:
        accGST[thisModel][itN] = [None] * nDataSplits
    # Repeat for the GFT error
    if doGFT:
        accGFT[itN] = [None] * nDataSplits
    # And using the data straight away
    if doData:
        accData[itN] = [None] * nDataSplits

    if doInterferometric:
        accInterferometric[itN] = [None] * nDataSplits
        
    #%%##################################################################
    #                                                                   #
    #                    DATA SPLIT REALIZATION                         #
    #                                                                   #
    #####################################################################
    
    # Start generating a new data split for each of the number of data splits
    # that we previously specified
    
    for split in range(nDataSplits):
        
        #%%##################################################################
        #                                                                   #
        #                    DATA HANDLING                                  #
        #                                                                   #
        #####################################################################
    
        ############
        # DATASETS #
        ############
        
        if doPrint:
            print("\tLoading data", end = '')
            if nDataSplits > 1:
                print(" for split %d" % (split+1), end = '')
            print("...", end = ' ', flush = True)
    
        #   Load the data, which will give a specific split
        data = Utils.dataTools.Authorship(authorName,thisRatioTrain,ratioValid,
                                          dataPath, graphNormalizationType,
                                          keepIsolatedNodes, forceUndirected,
                                          forceConnected)
        
        if doPrint:
            print("OK")

        #########
        # GRAPH #
        #########
        
        if doPrint:
            print("\tSetting up the graph...", end = ' ', flush = True)
    
        # Create graph
        adjacencyMatrix = data.getGraph()
        G = graphTools.Graph('adjacency',
                             adjacencyMatrix.shape[0],
                             {'adjacencyMatrix': adjacencyMatrix})
        G.computeGFT() # Compute the GFT of the stored GSO
    
        # And re-update the number of nodes for changes in the graph (due to
        # enforced connectedness, for instance)
        nNodes = G.N
        
        if doPrint:
            print("OK")

        #%%##################################################################
        #                                                                   #
        #                    GRAPH SCATTERING MODELS                        #
        #                                                                   #
        #####################################################################
        
        # Now that we have created the graph, we can build the graph scattering
        # models.
    
        modelsGST = {} # Store each model as a key in this dictionary, then we
            # can compute the output for each model inside a for (iterating over
            # the key), since all models have a computeTransform() method.
    
        if doDiffusion:
            modelsGST[diffusionName] = \
                              GST.DiffusionScattering(numScales, numLayers, G.W)
    
        if doMonicCubic:
            modelsGST[monicCubicName] = GST.MonicCubic(numScales,numLayers,G.W)
    
        if doTightHann:
            modelsGST[tightHannName] = GST.TightHann(numScales, numLayers, G.W)
    
        # Note that monic cubic polynomials and tight Hann's wavelets have other
        # parameters that are being set by default to the values in the
        # respective papers.
    
        # We want to determine which eigenbasis to use for the GFT. We try to
        # use the Laplacian since it's the same used in the wavelet cases, and
        # seems to be the one holding more "interpretability". If the Laplacian
        # doesn't exist (which could happen if the graph is directed or has
        # negative edge weights), then we use the eigenbasis of the adjacency.
        if doGFT:
            if G.L is not None:
                S = G.L
                if normalizeGSOforGFT:
                    S = graphTools.normalizeLaplacian(S)
                _, GFT = graphTools.computeGFT(S, order = 'increasing')
            else:
                S = G.W
                if normalizeGSOforGFT:
                    S = graphTools.normalizeAdjacency(S)
                _, GFT = graphTools.computeGFT(S, order = 'totalVariation')
    
            # For fair comparison, we might not want to use a representation
            # of different size (this means that we might want to use a number
            # of GFT coefficients equal to the dimension of the GST)
            if fullGFT:
                GFT = GFT.conj()
            else:
                if GFTfilterType == 'low':
                    startFreq = 0
                    endFreq = nGFTcoeff
                elif GFTfilterType == 'band':
                    startFreq = (GFT.shape[1] - nGFTcoeff)//2
                elif GFTfilterType == 'high':
                    startFreq = GFT.shape[1] - nGFTcoeff
                endFreq = startFreq + nGFTcoeff
                GFT = GFT[:,startFreq:endFreq].conj()
            
        #%%##################################################################
        #                                                                   #
        #                    CLASSIFIER: Linear SVM                         #
        #                                                                   #
        #####################################################################
        
        classifiers = {}
        
        ############
        # GET DATA #
        ############
        
        # Obs.: Data has nSamples x nNodes dimension, but the GST works assuming
        # a feature dimension in axis = 1, so we add it.
        xTrain, yTrain = data.getSamples('train')
        xTrain = xTrain.reshape(data.nTrain, 1, nNodes)
        xValid, yValid = data.getSamples('valid')
        xValid = xValid.reshape(data.nValid, 1, nNodes)
        xTest, yTest = data.getSamples('test')
        xTest = xTest.reshape(data.nTest, 1, nNodes)
        
         ##################################
        #                                  #
        #    GRAPH SCATTERING TRANSFORM    #
        #                                  #
         ##################################
        
        for thisModel in modelList:
            
            ################
            # ARCHITECTURE #
            ################
            
            classifiers[thisModel] = LinearSVC()
            
            ############
            # TRAINING #
            ############
            
            xTrainGST = modelsGST[thisModel].computeTransform(xTrain)
            xTrainGST = xTrainGST.squeeze(1) # Get rid of the intrinsic feature
                # dimension (axis = 1 dimension that was added before)
            classifiers[thisModel].fit(xTrainGST, yTrain)
            
            ##############
            # VALIDATION #
            ##############
            
            xValidGST = modelsGST[thisModel].computeTransform(xValid)
            xValidGST = xValidGST.squeeze(1)
            yHatValid = classifiers[thisModel].predict(xValidGST)
            # Compute accuracy:
            accValid = np.sum(yValid == yHatValid)/data.nValid
            
            if doPrint:
                print("\t%15s: %.4f" % (thisModel, accValid))
            
            ##############
            # EVALUATION #
            ##############
            
            xTestGST = modelsGST[thisModel].computeTransform(xTest)
            xTestGST = xTestGST.squeeze(1)
            yHatTest = classifiers[thisModel].predict(xTestGST)
            # Compute accuracy:
            accTest = np.sum(yTest == yHatTest)/data.nTest
            accGST[thisModel][itN][split] = accTest
            
         ###############################
        #                               #
        #    GRAPH FOURIER TRANSFORM    #
        #                               #
         ###############################
            
        if doGFT:
            
            ################
            # ARCHITECTURE #
            ################
            
            classifierGFT = LinearSVC()
            
            ############
            # TRAINING #
            ############
            
            xTrainGFT = (xTrain @ GFT).squeeze(1)
            classifierGFT.fit(xTrainGFT, yTrain)
            
            ##############
            # VALIDATION #
            ##############
            
            xValidGFT = (xValid @ GFT).squeeze(1)
            yHatValid = classifierGFT.predict(xValidGFT)
            # Compute accuracy:
            accValid = np.sum(yValid == yHatValid)/data.nValid
            
            if doPrint:
                print("\t%15s: %.4f" % (GFTname, accValid))
            
            ##############
            # EVALUATION #
            ##############
            
            xTestGFT = (xTest @ GFT).squeeze(1)
            yHatTest = classifierGFT.predict(xTestGFT)
            # Compute accuracy:
            accTest = np.sum(yTest == yHatTest)/data.nTest
            accGFT[itN][split] = accTest
            
        ######################
        #                     #
        #    STRAIGHT DATA    #
        #                     #
         #####################
            
        if doData:
            
            ################
            # ARCHITECTURE #
            ################
            
            classifierData = LinearSVC()
            
            ############
            # TRAINING #
            ############
            
            xTrainData = xTrain.squeeze(1)
            classifierData.fit(xTrainData, yTrain)
            
            ##############
            # VALIDATION #
            ##############
            
            xValidData = xValid.squeeze(1)
            yHatValid = classifierData.predict(xValidData)
            # Compute accuracy:
            accValid = np.sum(yValid == yHatValid)/data.nValid
            
            if doPrint:
                print("\t%15s: %.4f" % (DataName, accValid))
            
            ##############
            # EVALUATION #
            ##############
            
            xTestData = xTest.squeeze(1)
            yHatTest = classifierData.predict(xTestData)
            # Compute accuracy:
            accTest = np.sum(yTest == yHatTest)/data.nTest
            accData[itN][split] = accTest

        if doInterferometric:
            ######################
            #                     #
            #    Interferometric DATA    #
            #                     #
            #####################

            ################
            # ARCHITECTURE #
            ################

            classifierData = LinearSVC()

            ############
            # TRAINING #
            ############

            xTrainData = xTrain  # .squeeze(1)  # nTrain x nFeatures
            # print(np.linalg.norm(G.W))
            net = compute_representation(G.W.astype(np.float32), xTrainData.astype(np.float32),K=5,epochs=2)
            xTrainData = compute_feature(net, xTrainData.astype(np.float32),no_averaging=True)
            classifierData.fit(xTrainData, yTrain)

            ##############
            # VALIDATION #
            ##############

            # xValidData = xValid.squeeze(1)  # nValid x nFeatures
            xValidData = compute_feature(net, xValid.astype(np.float32),no_averaging=True)
            yHatValid = classifierData.predict(xValidData)
            # Compute accuracy:
            accValid = np.sum(yValid == yHatValid) / data.nValid

            if doPrint:
                print("\t%15s: %.4f" % ('Interferometric', accValid))

            ##############
            # EVALUATION #
            ##############

            xTestData = compute_feature(net, xTest.astype(np.float32),no_averaging=True)
            yHatTest = classifierData.predict(xTestData)
            # Compute accuracy:
            accTest = np.sum(yTest == yHatTest) / data.nTest
            accInterferometric[itN][split] = accTest


#%%##################################################################
#                                                                   #
#                       RESULTS (FIGURES)                           #
#                                                                   #
#####################################################################

# Now that we have computed the representation error of all runs, we can obtain
# a final result (mean and standard deviation) and plot it

meanAccGST = {} # Average across all graph realizations
stdDevAccGST = {} # Standard deviation across all graph realizations

######################
# COMPUTE STATISTICS #
######################

# Compute for each model
for thisModel in modelList:
    # Convert the lists into a matrix (2 dimensions):
    #  len(ratioTrain) x nDataSplits
    accGST[thisModel] = np.array(accGST[thisModel])

    # Compute mean and standard deviation across data splits
    meanAccGST[thisModel] = np.mean(accGST[thisModel], axis = 1)
    stdDevAccGST[thisModel] = np.std(accGST[thisModel], axis = 1)

# Compute for GFT
if doGFT:
    # Convert the lists into a matrix (2 dimensions):
    #  len(ratioTrain) x nDataSplits
    accGFT = np.array(accGFT)

    # Compute mean and standard deviation across data splits
    meanAccGFT = np.mean(accGFT, axis = 1)
    stdDevAccGFT = np.std(accGFT, axis = 1)

# Compute for Data
if doData:
    # Convert the lists into a matrix (2 dimensions):
    #  len(ratioTrain) x nDataSplits
    accData = np.array(accData)

    # Compute mean and standard deviation across data splits
    meanAccData = np.mean(accData, axis = 1)
    stdDevAccData = np.std(accData, axis = 1)

if doInterferometric:
    # Convert the lists into a matrix (2 dimensions):
    #  len(ratioTrain) x nDataSplits
    accInterferometric = np.array(accInterferometric)

    # Compute mean and standard deviation across data splits
    meanAccInterferometric = np.mean(accInterferometric, axis=1)
    stdDevAccInterferometric = np.std(accInterferometric, axis=1)

################
# SAVE RESULTS #
################

# If we're going to save the results (either figures or pickled variables) we
# need to create the directory where to save them

if doSaveVars or doFigs:
    saveDirResults = os.path.join(saveDir,'results')
    if not os.path.exists(saveDirResults):
        os.makedirs(saveDirResults)

##################
# SAVE VARIABLES #
##################

if doSaveVars:
    # Save all these results that we use to reconstruct the values
    #   Save these variables
    varsDict = {}
    varsDict['accGST'] = accGST
    varsDict['meanAccGST'] = meanAccGST
    varsDict['stdDevAccGST'] = stdDevAccGST
    if doGFT:
        varsDict['accGFT'] = accGFT
        varsDict['meanAccGFT'] = meanAccGFT
        varsDict['stdDevAccGFT'] = stdDevAccGFT
    if doData:
        varsDict['accData'] = accData
        varsDict['meanAccData'] = meanAccData
        varsDict['stdDevAccData'] = stdDevAccData

    if doInterferometric:
        varsDict['accInterferometric'] = accInterferometric
        varsDict['meanAccInterferometric'] = meanAccInterferometric
        varsDict['stdDevAccInterferometric'] = stdDevAccInterferometric
    #   Determine filename to save them into
    varsFilename = 'classificationAccuracy.pkl'
    pathToFile = os.path.join(saveDirResults, varsFilename)
    with open(pathToFile, 'wb') as varsFile:
        pickle.dump(varsDict, varsFile)

#########
# PLOTS #
#########

if doFigs:
    # Create figure handle
    accFig = plt.figure(figsize = (1.61*figSize, 1*figSize))
    nTotal = data.nTest + data.nTrain + data.nValid
    nTrain = np.round(ratioTrain * nTotal)
    # For each model, plot the results
    for thisModel in modelList:
        plt.errorbar(nTrain, meanAccGST[thisModel],
                     yerr = stdDevAccGST[thisModel],
                     linewidth = lineWidth, marker = markerShape,
                     markersize = markerSize)
    # If there's representation error of the GFT, plot it
    if doGFT:
        plt.errorbar(nTrain, meanAccGFT, yerr = stdDevAccGFT,
                     linewidth = lineWidth, marker = markerShape,
                     markersize = markerSize)
    # If there's a bound, plot it
    if doData:
        plt.errorbar(nTrain, meanAccData, yerr = stdDevAccData,
                     linewidth = lineWidth,
                     marker = markerShape, markerSize = markerSize)

    if doInterferometric:
        plt.errorbar(nTrain, meanAccInterferometric, yerr=stdDevAccInterferometric,
                     linewidth=lineWidth,
                     marker=markerShape, markerSize=markerSize)

    plt.ylabel(r'Classification accuracy')
    plt.xlabel(r'Number of training samples')
    # Add the names to the legends
    if doGFT:
        modelList.append(GFTname)
    if doData:
        modelList.append(DataName)
    if doInterferometric:
        modelList.append('IGT')
    plt.legend(modelList)
    accFig.savefig(os.path.join(saveDirResults, 'classifAccFig.pdf'),
                         bbox_inches = 'tight')