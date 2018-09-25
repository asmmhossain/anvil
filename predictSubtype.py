#!/usr/bin/env python3

'''
    predictSubtype.py
    A python program to predict subtype of a query

    Three options are used:
        - predict query subtype in the forward direction
        - predict query subtype in the reverse direction
        - predict query subtype using both forward and reverse direction

    This program takes as input the following:
        - A FASTA formatted file containing query sequences
        - number of residues for the context; context size
        - window size for decision tree
        - step size for the sliding window
        - name of the output file
        - threshold for likelihood difference in decision tree
        - directory that holds NN parameter files
'''

#***********************************************
from Bio import SeqIO
import os, sys, argparse
import numpy as np
import subNet2 as subnet
import random
import math
import json
import subprocess, time
from joblib import Parallel, delayed
from joblib.pool import has_shareable_memory

#*************************************************************
# define the array represntation of the nucleotide alphabets
dna = {'A' : [1,0,0,0],
       'C' : [0,1,0,0],
       'G' : [0,0,1,0],
       'T' : [0,0,0,1],
       'M' : [0.5,0.5,0,0],
       'R' : [0.5,0,0.5,0],
       'W' : [0.5,0,0,0.5],
       'S' : [0,0.5,0.5,0],
       'Y' : [0,0.5,0,0.5],
       'K' : [0,0,0.5,0.5],
       'V' : [0.33,0.33,0.33,0],
       'H' : [0.33,0.33,0,0.33],
       'D' : [0.33,0,0.33,0.33],
       'B' : [0,0.33,0.33,0.33],
       'N' : [0.25,0.25,0.25,0.25]
     }

#*************************************************************
# define the positions of the output nucleotides
dPos = {'A' : 0,
        'C' : 1,
        'G' : 2,
        'T' : 3
      }

##********************************************************##
def validateInput(x):
    try:
        x = int(x)
    except ValueError as e:
        sys.exit(e)

    if x <= 0:
        parser.error('Invalid input value %r, must be a positive integer' % (x,))

    return x

#*************************************************************
def padseq(s,n):
    '''adds n Ns at the beginning of the sequence'''
    return('N'*n+s)


#*******************************************************
def getNonRedundantListOrder(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]

#************************************************************
def seq2array(seq):
    '''
    This function converts a sequence into corresponding array represntation
    based on 'dna' dictionary
    '''
    #print(seq)
    lst = list()
    for s in seq:
        lst.extend(dna.get(s,[0,0,0,0]))
    return lst

#********************************************************
# this is to check duplicate values for the same argument
class UniqueStore(argparse.Action):
    def __call__(self, parser, namespace, values, option_string):
        if getattr(namespace, self.dest, self.default) is not None:
            #print(option_string+values)
            parser.error(option_string + " appears several times.")
        setattr(namespace, self.dest, values)

#************************************************************************
def validateCPU(cpu):
    '''
    checks number of available processors and sets value
    '''
    try:
        n = int(cpu)
    except ValueError:
        parser.error('Invalid input value %r, must be a positive integer' % (cpu,))

    numThreads = os.cpu_count()

    if numThreads == None:
        return 1
    elif numThreads >= n:
        return cpu
    elif numThreads < n:
        return numThreads

#***********************************************************
def loadNetworks(paramDir):
    '''
        Loads parameter files in the forward direction
    '''

    # get the names of the subtypes present in the parameter directory
    forward_param_files = sorted([f for f in os.listdir(paramDir) if f.endswith('.param')])

    # check parameter files are in pairs
    forward_subtypes = [pf.split('.')[0] for pf in forward_param_files]

    subtypes = list(forward_subtypes)
    #print(subtypes)

    ### Load the neural netwrok parameters
    forwardNetworks = dict()

    for stp in subtypes:
        sfName = os.path.join(paramDir, stp.upper() + '.param')
        if os.path.exists(sfName) and os.stat(sfName).st_size != 0:
            tNet = subnet.load(sfName)
            forwardNetworks[stp] = tNet
        else:
            subtypes.remove(stp)
            continue

    return forwardNetworks, subtypes

#***********************************************************
def loadNeuralNetworks(paramDir):
    '''
        Checks whether parameter files are present in pairs
        Loads the parameters in two dictionaries: forward and reverese
        The dictionaries represent the forward_NN and the reverse_NN
    '''
    # get the names of the subtypes present in the parameter directory
    forward_param_files = sorted([f for f in os.listdir(paramDir) if f.endswith('.forward.param')])
    reverse_param_files = sorted([f for f in os.listdir(paramDir) if f.endswith('.reverse.param')])
    # check parameter files are in pairs
    forward_subtypes = [pf.split('.')[0] for pf in forward_param_files]
    reverse_subtypes = [pf.split('.')[0] for pf in reverse_param_files]

    # are there any missing parameter pairs: only present in one direction???
    missing = set(forward_subtypes) ^ set(reverse_subtypes)

    if missing:
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' ERROR: Subtypes found with missing pair of the parameter files:'
        msg += ' {}. Please check the parameter files in {} and run again.\n'.format(' '.join(missing),paramDir)
        sys.exit(msg)

    subtypes = list(forward_subtypes)
    #print(subtypes)

    ### Load the neural netwrok parameters in two dictionaries
    # these dictionaries will represent the NNs in forward and reverse direction
    forwardNetworks = dict()

    reverseNetworks = dict()

    for stp in subtypes:
        sfName = os.path.join(paramDir, stp.upper() + '.forward.param')
        if os.path.exists(sfName) and os.stat(sfName).st_size != 0:
            tNet = subnet.load(sfName)
            forwardNetworks[stp] = tNet
        else:
            subtypes.remove(stp)
            continue
        # reverse
        sfName = os.path.join(paramDir, stp.upper() + '.reverse.param')
        if os.path.exists(sfName) and os.stat(sfName).st_size != 0:
            tNet = subnet.load(sfName)
            reverseNetworks[stp] = tNet
        else:
            subtypes.remove(stp)

    # are there any unloaded parameter pairs: only present in one direction???
    missingNN = set(forwardNetworks.keys()) ^ set(reverseNetworks.keys())

    if set(forwardNetworks.keys()) != set(reverseNetworks.keys()):
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' ERROR: Subtypes found where one of the parameter files did not load:'
        msg += ' {}. Please check the parameter files in {} and run again.\n'.format(' '.join(missingNN),paramDir)
        sys.exit(msg)

    else:
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' Neural networks loaded for ANVIL. \n'
        print(msg)


    return forwardNetworks, reverseNetworks, subtypes

#***************************************************
def readQuerySequences(args):
    '''
        Checks whether the query file contains valid FASTA formatted queries
        returns a list of sequences
    '''
    ## read in the query sequences
    try:
        qSeqs = list(SeqIO.parse(args.query,'fasta'))
    except FileNotFoundError as e:
        eMsg = '\nThe query sequence file <{}> could not be found.'.format(args.query)
        eMsg += ' Please try again with correct sequence file name.\n'
        sys.exit(eMsg)

    ## check if the sequences were read properly
    if len(qSeqs) == 0:
        msg = 'Query sequences were not read properly'
        msg += '\nPlease run again with valid FASTA sequence file with at least one sequence\n'
        sys.exit(msg)
    else:
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' Read {} sequences successfully for subtype prediction\n'.format(len(qSeqs))
        print(msg)

    return qSeqs

#*******************************************************
def challenge(sLike,target,nSubtypes,start,end,thr):
    '''This function computes the sum of likelihoods for each subtypes
    in the given window and finds the most likely subtype
    '''

    # get the sum of likelihoods for each subtypes
    #sumLL = [sum(sLike[i][start:end])for i in range(nSubtypes)]
    sumLL = np.sum(sLike[:,start:end],axis=1)

    # find the maximum likelihood and index
    #maxLL = max(sumLL)
    maxLL = np.amax(sumLL)

    #maxIndex = sumLL.index(maxLL)
    maxIndex = np.argmax(sumLL)

    # return best matching subtype according to COMET's decision tree
    if (maxLL - sumLL[target]) <= thr:
        return target
    else:
        return maxIndex

#*******************************************************
def getNonRedundantListOrder(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]

#***********************************************************
def getProbability(stpNet,sequence,cSize):
    '''Returns a list of likelihood values calculated from the feedforward network of a subtype'''

    padSeq = padseq(sequence,cSize)
    #sys.exit(padSeq)

    sLen = len(padSeq)

    probs = [0.0]*sLen

    # calculate the input size for ANN
    inSize = cSize * 4

    # call feedforward() for each of the nucleotides
    for i in range(cSize,len(padSeq)):
        start = i - cSize
        ctx = padSeq[start:i]
        ch = padSeq[i]
        if set(ctx).issubset('ACGTN') and ch in 'ACGT':
            # get the softmax likelihoods
            chML = stpNet.feedforward(np.reshape(seq2array(ctx),(inSize,1)))
            #print(ctx,ch,chML)

            # get the probability of the character
            chProb = chML[dPos.get(ch)][0]
            #print(ctx,ch,chProb)

            # add the probability of 'ch' to the list 'probs'
            probs[i] = chProb
            #sys.exit()

    return probs[cSize:]


#*******************************************************
def detectBreakpoints(assignedSubtypes,subtypes,fNetwork,qSeq,qId,wSize,args,zName,direction):


    qLen = len(qSeq)

    # get number of assigned subtypes
    nParents = len(assignedSubtypes)

    # create a matrix to hold the probability values
    pMatrix = np.zeros((nParents,qLen))

    # create a numpy matrix to hold the average probability  values
    aMatrix = np.zeros((nParents,qLen))

    cSize = args.context

    #print(assignedSubtypes)

    #print(cSize,nParents,qLen)

    # generate the probability matrix
    for r in range(nParents):
       pMatrix[r] = getProbability(fNetwork[subtypes[assignedSubtypes[r]]],qSeq,cSize)  #(stpNet,sequence,cSize)


    # calculate average scores at each position for each assigned subtypes
    # for insufficient values on both side, average will be calculated at
    # positions 7..(qLen-7)
    hwSize = wSize // 2
    lastPos = qLen - hwSize

    sName = os.path.join(zName, qId + '.' + direction + '.prob.txt')
    pName = os.path.join(zName, qId + '.' + direction + '.prob.png')

    lName = os.path.join(zName, direction + '.breakpoints.log')
    lh = open(lName,'a')

    # create the probability distribution file
    fh = open(sName,'w')
    fh.write('Positions')

    for sub in assignedSubtypes:
        sType = subtypes[sub]
        fh.write('\t{}'.format(sType))
    #fh.write('\n')

    for i in range(qLen):
        fh.write('\n{}'.format(str(i)))

        # for all subtypes in assignedSubtypes
        for k in range(nParents):
            #print(k)
            #if i < hwSize:
                #start = 0
            #else:
                #start = i - hwSize

            start = 0 if i < hwSize else (i-hwSize)

            end = i + hwSize + 1

            tLL = pMatrix[k][start:end]
            aMatrix[k][i] = np.average(tLL)

            #if i < hwSize or i >= lastPos:
                #aMatrix[k][i] = pMatrix[k][i]
            #else:
                #start = i - hwSize
                #end = i + hwSize + 1
                #tLL =  pMatrix[k][start:end]
                #aMatrix[k][i] = np.average(tLL)

            fh.write('\t{}'.format(aMatrix[k][i]))

        #fh.write('\n')

    # create the probability distribution plot using R script
    cl = ['Rscript','analyseBreakpoints.R',sName,pName, qId]
    try:
        subprocess.check_call(cl,stdout=lh,stderr=lh)
    except subprocess.CalledProcessError as e:
        pass

    fh.close()



    #plt.plot(aMatrix[0])
    #plt.xlabel('Nucleotide positions')
    #plt.ylabel('Probability')
    #plt.show()
    return


#****************************************************
def check_subtype_single(sLike,query,seqId,subtypes,nSubtypes,fNetwork,qLen,args,zName):
    '''
        Uses COMET's desition tree (simplified) to call subtypes for a query
            sLike: likelihood matrix
            seqID: sequence identifier
            subtypes: list of subtype names
            nSubtypes: number of reference subtypes
            qLen: length of the query sequence
            args: command line arguments

    '''
    thr = args.thr
    # get the sum of likelihoods for all subtypes
    #sumLL = [sum(sLike[i]) for i in range(nSubtypes)]
    sumLL = np.sum(sLike,axis=1)
    #print(sumLL)

    # find the index of most likely subtype PURE/CRF
    #maxS = max(sumLL)
    maxS = np.amax(sumLL)

    #S = sumLL.index(maxS)
    S = np.argmax(sumLL)

    wSize = args.wSize # set window size
    bSize = args.bSize # set the step size

    # Pre-compute number of windows
    numOfWindows = int((qLen-wSize)/bSize)+1
    #print('numOfWindows',numOfWindows)

    # create a list of subtype assignment for each window
    subAssignment = [S]*numOfWindows
    iWindow = 0

    # get the most likely subtype for each window
    for i in range(0,numOfWindows*bSize,bSize):
        start = i
        end = i + wSize
        #print(iWindow)
        subAssignment[iWindow] = challenge(sLike,S,nSubtypes,start,end,thr)
        iWindow += 1

    assignedSubtypes = getNonRedundantListOrder(subAssignment)
    if len(assignedSubtypes) == 1:
        msg = seqId + '\t' + subtypes[S]
        #print(msg)
        return msg
    else:
        msg = seqId + '\t' + 'unassigned (potential recombinant)\t'
        for asub in assignedSubtypes:
            msg += subtypes[asub] + ' '
        #print(msg)

        if args.mode in [1,3]:
            detectBreakpoints(assignedSubtypes,subtypes,fNetwork,query,seqId,101,args,zName,'forward')

        if args.mode in [2,3]:
            queryReverse = query[::-1]
            detectBreakpoints(assignedSubtypes,subtypes,fNetwork,queryReverse,seqId,101,args,zName,'reverse')

        return msg

#***********************************************************
def getLikelihoods(stpNet,sequence,cSize):
    '''Returns a list of likelihood values calculated from the feedforward network of a subtype'''

    padSeq = padseq(sequence,cSize)
    #sys.exit(padSeq)

    sLen = len(padSeq)

    #likes = [0.0]*sLen
    likes = np.zeros(sLen)

    # calculate the input size for ANN
    inSize = cSize * 4

    # call feedforward() for each of the nucleotides
    for i in range(cSize,len(padSeq)):
        start = i - cSize
        ctx = padSeq[start:i]
        ch = padSeq[i]
        if set(ctx).issubset('ACGTN') and ch in 'ACGT':
            # get the softmax likelihoods
            chML = stpNet.feedforward(np.reshape(seq2array(ctx),(inSize,1)))
            #print(ctx,ch,chML)

            # get the probability of the character
            chProb = chML[dPos.get(ch)][0]
            #print(ctx,ch,chProb)

            # calculate the log likelihood of the probability
            chProb = math.log10(chProb)
            #print(ctx,ch,chProb)

            # add the likelihood of 'ch' to the list 'likes'
            #likes.append(chProb)
            likes[i] = chProb
            #sys.exit()

    return likes[cSize:]

#***************************************************************************
def runNeuralNetworkSingleSubtype(stpNet,sequence,likes,cSize,inSize,pos):
    '''
        This function will be called parallely
    '''
    start = pos - cSize
    ctx = sequence[start:pos]
    ch = sequence[pos]
    if set(ctx).issubset('ACGTN') and ch in 'ACGT':
        # get the softmax likelihoods
        chML = stpNet.feedforward(np.reshape(seq2array(ctx),(inSize,1)))
        #print(ctx,ch,chML)

        # get the probability of the character
        chProb = chML[dPos.get(ch)][0]
        #print(ctx,ch,chProb)

        # calculate the log likelihood of the probability
        chProb = math.log10(chProb)
        #print(ctx,ch,chProb)

        # add the likelihood of 'ch' to the list 'likes'
        #likes.append(chProb)
        likes[pos] = chProb
        #sys.exit()



#•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••••
def getLikelihoodsParallel(stpNet,sequence,cSize,args):
    '''
        Returns a list of likelihood values calculated from the feedforward network of a subtype
        Uses joblib to parallely calculate likelihoods of the residues
    '''

    padSeq = padseq(sequence,cSize)
    #sys.exit(padSeq)

    sLen = len(padSeq)

    #likes = [0.0]*sLen
    likes = np.zeros(sLen)

    # calculate the input size for ANN
    inSize = cSize * 4

    Parallel(n_jobs=args.cpu)(delayed(has_shareable_memory)\
        (runNeuralNetworkSingleSubtype(stpNet,padSeq,likes,cSize,inSize,i)) for i in range(cSize,len(padSeq)))

    return likes[cSize:]

#****************************************************************************
def getLikelihoodMatrix(qSeq,subtypes,nSubtypes,fNetwork,args):
    '''
    This function calculate the likelihoods of each nucleotide positions in a query
    generates a matrix of log-likelihood values for each of the subtypes
    '''

    # get reverse sequence
    qSeqReverse = qSeq[::-1]

    qLen = len(qSeq)

    # Create a list to hold likelihoods for all the nucleotide positions
    # each row represents likelihoods generated by each of the subtype PPMD models

    # matrix to store forward likelihoods
    fMatrix = np.zeros((nSubtypes,qLen))

    # matrix to store reverse likelihoods
    rMatrix = np.zeros((nSubtypes,qLen))

    # matrix to store combined likelihoods
    cMatrix = np.zeros((nSubtypes,qLen))

    # get the context size

    cSize = int(args.context)

    # generate likelihoods based on each subtype models
    # generate likelihoods in the forward direction
    if args.mode in [1,3]:
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' Calculating log-likelihoods in the forward direction.\n'
        print(msg)
        for r in range(nSubtypes):
            fMatrix[r] = getLikelihoods(fNetwork[subtypes[r]],qSeq,cSize) #(stpNet,sequence,cSize)
            #fMatrix[r] = getLikelihoodsParallel(fNetwork[subtypes[r]],qSeq,cSize,args) #(stpNet,sequence,cSize)

        cMatrix += fMatrix

    # generate likelihoods in the reverse direction
    if args.mode in [2,3]:
        msg = '\n[' + time.strftime('%d %b %H:%M:%S') + ']'
        msg += ' Calculating log-likelihoods in the reverse direction.\n'
        print(msg)
        for r in range(nSubtypes):
            fMatrix[r] = getLikelihoods(fNetwork[subtypes[r]],qSeqReverse,cSize) #(stpNet,sequence,cSize)

        cMatrix += rMatrix

    if args.mode == 3:
        cMatrix /= 2

    return cMatrix

    #return check_subtype(lMatrix,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName)
    #return check_subtype_single(lMatrix,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName)


##**************************************************
def getArguments():
    '''
        Parse all the command line arguments from the user
    '''

    parser = argparse.ArgumentParser(description='Predicts subtype of a sequence based on PPMD models trained using reference sequences', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-q','--query',required=True,help='Query sequence file in FASTA format for subtype classification',action=UniqueStore)
    parser.add_argument('-c','--context',type=int,default=8,help='Context size for the neural network (default: 8)')
    parser.add_argument('-w','--wSize',type=int,default=100,help='Window size for decision tree (default: 100)')
    parser.add_argument('-b','--bSize',type=int,default=3,help='Step size of the windows for decision tree (default: 3)',action='append')
    parser.add_argument('-o','--outFile',required=True,help='Output file for subtype prediction results',action=UniqueStore)
    parser.add_argument('-t','--thr',type=int,default=32,help='Threshold difference used in decision tree (default: 32)',action='append')
    parser.add_argument('-d','--dir',required=True,help='Directory for saved parameter files for neural networks',action=UniqueStore)
    parser.add_argument('-n', '--cpu', type=int, help="Number of CPUs to use (default: 1)")
    parser.add_argument('-m', '--mode', type=int, choices=[1,2,3], action=UniqueStore, \
                        help='Prediction mode can be Forward(1)/Reverse(2)/Both(3)')


    args = parser.parse_args()

    if args.context < 1:
        parser.error('Invalid input value %r, must be a positive integer' % (args.context,))

    if args.wSize < 1:
        parser.error('Invalid input value %r, must be a positive integer' % (args.wSize,))

    if args.bSize < 1:
        parser.error('Invalid input value %r, must be a positive integer' % (args.bSize,))

    if args.thr < 1:
        parser.error('Invalid input value %r, must be a positive integer' % (args.thr,))

    if not args.cpu:
        args.cpu = 1

    if not args.mode:
        args.mode = 1

    return args

def runAnvil(q,subtypes,numSubtypes,fNetwork,args,zName):
    '''
    '''
    query = str(q.seq).upper().replace('-','')
    likelihoodMatrix = getLikelihoodMatrix(query,subtypes,numSubtypes,fNetwork,args)
    result = check_subtype_single(likelihoodMatrix,query,q.id,subtypes,numSubtypes,fNetwork,len(query),args,zName)
    print(result)

#***********************************************************
if __name__=="__main__":
    args = getArguments()

    ## get context size
    cSize = int(args.context)

    # get the directory name where all the neural networks are saved
    paramDir = args.dir

    # load the neural networks from the parameter directory <paramDir>
    #fNetwork, rNetwork, subtypes = loadNeuralNetworks(paramDir)
    fNetwork, subtypes = loadNetworks(paramDir)
    # get number of subtypes
    numSubtypes = len(subtypes)
    #print(numSubtypes)

    querySeqs = readQuerySequences(args)

    # Create a directory to stote the output of the ANN_subtyping run
    timeNow = time.strftime('%Y-%m-%d-%H%M%S')
    zName = 'anvil.' + timeNow
    #print(zName)
    zName = 'test_anvil'
    '''
    try:
        os.mkdir(zName)
        #pass

    except OSError as e:
        sys.exit(e)
    '''

    oName = os.path.join(zName, args.outFile)

    #sys.exit()
    with open(oName,'w') as oh:
        oh.write('Name\tSubtype\n')

    #Parallel(n_jobs=args.cpu)(delayed(runAnvil)(query,subtypes,numSubtypes,fNetwork,args,zName) for query in querySeqs[:10])

    #'''
    for q in querySeqs[:10]:
        query = str(q.seq).upper().replace('-','')
        likelihoodMatrix = getLikelihoodMatrix(query,subtypes,numSubtypes,fNetwork,args)
        result = check_subtype_single(likelihoodMatrix,query,q.id,subtypes,numSubtypes,fNetwork,len(query),args,zName)
        print(result)
        with open(oName,'a') as oh:
            oh.write('{}\n'.format(result))
    #'''
