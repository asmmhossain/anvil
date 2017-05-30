from Bio import SeqIO
#import umsgpack
import os, sys, argparse
import numpy as np
import subNet2 as subnet
import random
import math
import json

import subprocess, time


'''
This program will take a FASTA file with query sequences.
For each query:
    - Log-likelihoods of all the nucleotide positions will be calculated for each subtypes using neural networks
    - Most likely subtype will be identified
    - The likelihoods of the most likely subtype will be challenged by others using sliding windows
    - For potential recombinants, breakpoint analysis will be performed

'''

#***************************************
#newData = True
#sType = 'B'
#refDir = 'ANN_prrt_8_500'

#cSize = 8
#inSize = cSize * 4
#vocab = 'ACGT'
#***************************************

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

#*************************************************************
# define the positions of the output nucleotides
dPos = {'A' : 0,
        'C' : 1,
        'G' : 2,
        'T' : 3
      }

#*************************************************************

##********************************************************##
def contextThreshold(x):
    try:
        x = int(x)
    except ValueError as e:
        sys.exit(e)
    
    if x <= 0:
        raise argparse.ArgumentTypeError('%r must be a positive integer' % (x,)) 
    
    return x   
#*******************************************************

#*************************************************************
def padseq(s,n):
    '''adds n Ns at the beginning of the sequence'''
    return('N'*n+s)

#************************************************************

#*******************************************************
def getNonRedundantListOrder(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]
#*******************************************************    

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
#***********************************************************

#***********************************************************
def getLikelihoods(stpNet,sequence,cSize):
    '''Returns a list of likelihood values calculated from the feedforward network of a subtype'''
    
    padSeq = padseq(sequence,cSize)
    #sys.exit(padSeq)
    
    sLen = len(padSeq)
    
    likes = [0.0]*sLen
    
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

#***********************************************************

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

#***********************************************************

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

#****************************************************
def challengeType(sLike,target,start,end,flag,subtypes):
    '''
        calculates sum of log likelihood for each reference in the window
        if other - PS > 28 for all others returns 'True', else returns 'False'
    '''
    
    psLike = sLike[target][start:end]
    
    for i in range(len(sLike)):
        #if ppmd[i]['type'] == ppmd[target]['type']:
            #continue
        oLike = sLike[i][start:end]
                
        diff = sum(oLike) - sum(psLike)
        
        if diff > 46:
            if flag:
                print('challenge: false', subtypes[i], subtypes[target], diff)
            return False
        
    return True
#****************************************************

#*******************************************************
def detectBreakpoints(assignedSubtypes,subtypes,networks,query,wSize,args,zName):

    
    # convert sequences into upper case and remove gaps
    qSeq = str(query.seq).upper().replace('-','')   
    
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
       pMatrix[r] = getProbability(networks[subtypes[assignedSubtypes[r]]],qSeq,cSize)  #(stpNet,sequence,cSize)
           
    #print(query.id,'\n',pMatrix[:,10:15])
    
    #sys.exit()     
    
    # calculate average scores at each position for each assigned subtypes
    # for insufficient values on both side, average will be calculated at 
    # positions 7..(qLen-7)
    hwSize = wSize // 2
    lastPos = qLen - hwSize
    
    sName = zName + '/' + query.id + '.prob.txt'
    pName = zName + '/' + query.id + '.prob.png'
    
    lName = zName + '/' + 'breakpoints.log'
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
    cl = ['Rscript','analyseBreakpoints.R',sName,pName, query.id]
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
    
#*******************************************************

#****************************************************
def check_subtype_single(sLike,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName):
    '''
        Uses COMET's desition tree (simplified) )to call subtypes for a query
            sLike: likelihood matrix
            seqID: sequence identifier
            subtypes: list of subtype names
            nSubtypes: number of reference subtypes
            qLen: length of the query sequence
            args: command line arguments
        
    '''
    # get the sequence ID
    seqId = query.id
    
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
    
    ## find the most likely PURE subtype
    ##maxPS = max(sumLL[pIndex:])
    ##maxPS = np.amax(sumLL[pIndex:])
    ##PS = (sumLL[pIndex:].index(maxPS)) + pIndex
    ##PS = np.argmax(sumLL[pIndex:]) + pIndex
    
    #print(seqId, subtypes[S],maxS,subtypes[PS],maxPS)
    
    #sys.exit()
    #with open(args.outFile,'a') as fh:
        #fh.write('{}\t{}\t{}\t{}\t{}\n'.format(seqId, subtypes[S],maxS,subtypes[PS],maxPS)) 
    #return
    
    wSize = args.wSize # set window size
    bSize = args.bSize # set the step size

    # Pre-compute number of windows
    numOfWindows = int((qLen-wSize)/bSize)+1
    #print('numOfWindows',numOfWindows)
    
    # check the PURE subtype first
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
        msg = seqId + '\t' + 'unassigned_1\t' 
        for asub in assignedSubtypes:
            msg += subtypes[asub] + ' '
        #print(msg)
        detectBreakpoints(assignedSubtypes,subtypes,networks,query,101,args,zName)
        return msg    
    
#****************************************************            

#****************************************************
def check_subtype(sLike,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName):
    '''
        Uses COMET's desition tree to call subtypes for a query
            sLike: likelihood matrix
            seqID: sequence identifier
            subtypes: list of subtype names
            nSubtypes: number of reference subtypes
            qLen: length of the query sequence
            args: command line arguments
        
    '''
    # get the sequence ID
    seqId = query.id
    
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
    
    # find the most likely PURE subtype
    #maxPS = max(sumLL[pIndex:])
    maxPS = np.amax(sumLL[pIndex:])
    #PS = (sumLL[pIndex:].index(maxPS)) + pIndex
    PS = np.argmax(sumLL[pIndex:]) + pIndex
    
    #print(seqId, subtypes[S],maxS,subtypes[PS],maxPS)
    
    #sys.exit()
    #with open(args.outFile,'a') as fh:
        #fh.write('{}\t{}\t{}\t{}\t{}\n'.format(seqId, subtypes[S],maxS,subtypes[PS],maxPS)) 
    #return
    
    wSize = args.wSize # set window size
    bSize = args.bSize # set the step size

    # Pre-compute number of windows
    numOfWindows = int((qLen-wSize)/bSize)+1
    #print('numOfWindows',numOfWindows)
    
    # check the PURE subtype first
    # create a list of subtype assignment for each window
    subAssignment = [PS]*numOfWindows
    iWindow = 0    

    # get the most likely subtype for each window
    for i in range(0,numOfWindows*bSize,bSize):
        start = i
        end = i + wSize
        #print(iWindow)
        subAssignment[iWindow] = challenge(sLike,PS,nSubtypes,start,end,thr)
        iWindow += 1
        
    
    if S == PS:
        assignedSubtypes = getNonRedundantListOrder(subAssignment)
        if len(assignedSubtypes) == 1:
            msg = seqId + '\t' + subtypes[PS] + '\t(PURE)'
            #print(msg)
            return msg
        else:
            msg = seqId + '\t' + 'unassigned_1\t' 
            for asub in assignedSubtypes:
                msg += subtypes[asub] + ' '
            #print(msg)
            detectBreakpoints(assignedSubtypes,subtypes,networks,query,101,args,zName)
            return msg
    else: # S != PS
        assignedSubtypes = getNonRedundantListOrder(subAssignment)
        if len(assignedSubtypes) == 1:
            msg = seqId + '\t' + subtypes[PS] + '\t(Check ' + subtypes[S] + ')'
            #print(msg)
            return msg
        else: # needs checking CRF
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
                msg = seqId + '\t' + subtypes[S] + '\t(CRF)'
                #print(msg)
                return msg
            else:
                msg = seqId + '\t' + 'unassigned_2\t' 
                for asub in assignedSubtypes:
                    msg += subtypes[asub] + ' '
                #print(msg)
                detectBreakpoints(assignedSubtypes,subtypes,networks,query,101,args,zName)
                return msg
#****************************************************            


#****************************************************************************
def calculateLikelihood(query,subtypes,nSubtypes,networks,pIndex,args,zName):
    '''
    This function calculate the likelihoods of each nucleotide positions in a query
    generates a matrix of log-likelihood values for each of the subtypes
    '''
    #print('inside calculateLikelihood')
    # convert sequences into upper case and remove gaps
    qSeq = str(query.seq).upper().replace('-','')

    qLen = len(qSeq)

    # Create a list to hold likelihoods for all the nucleotide positions
    # each row represents likelihoods generated by each of the subtype PPMD models

    #lMatrix = [[]]*nSubtypes
    lMatrix = np.zeros((nSubtypes,qLen))
    
    # get the context size
    
    cSize = int(args.context)

    # generate likelihoods based on each subtype models
    for r in range(nSubtypes):
        lMatrix[r] = getLikelihoods(networks[subtypes[r]],qSeq,cSize) #(stpNet,sequence,cSize)

    #print(lMatrix[0][5:15])
    #print(len(lMatrix))
    
    #sys.exit()
    
    #return check_subtype(lMatrix,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName)
    return check_subtype_single(lMatrix,query,subtypes,nSubtypes,networks,qLen,pIndex,args,zName)
    

##**************************************************    

def getArguments():
    '''
        Parse all the command line arguments from the user
    '''
    
    parser = argparse.ArgumentParser(description='Predicts subtype of a sequence based on PPMD models trained using reference sequences', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-q','--query',required=True,help='Query sequence file in FASTA format for subtype classification')
    parser.add_argument('-c','--context',type=contextThreshold,default=8,help='Context size for the neural network (default: 8)')
    parser.add_argument('-w','--wSize',type=contextThreshold,default=100,help='Window size for decision tree (default: 100)')
    parser.add_argument('-b','--bSize',type=contextThreshold,default=3,help='Step size of the windows for decision tree (default: 3)')
    parser.add_argument('-o','--outFile',required=True,help='Output file for subtype prediction results')
    parser.add_argument('-t','--thr',type=contextThreshold,default=28,help='Threshold difference used in decision tree (default: 28)')
    parser.add_argument('-d','--dir',required=True,help='Directory for saved parameter files for neural networks')    


    args = parser.parse_args()
    
    return args
    
##********************************************************##

#***********************************************************

if __name__=="__main__":
    args = getArguments()

    ## get context size
    cSize = int(args.context)
     
    # get the directory name where all the neural networks are saved
    pDir = args.dir
    
    # get the names of the subtypes present in the parameter directory
    param_files = sorted([f for f in os.listdir(pDir) if f.endswith('.param')])
    subtypes = [pf.split('.')[0] for pf in param_files]
    #print(subtypes)
    #sys.exit()
               
    ### Load all the trained ANNs in a dictionary of networks
    # create an empty dictionary
    networks = dict()

    for stp in subtypes:
        sfName = pDir + '/' + stp.upper() + '.param'
        if os.path.exists(sfName) and os.stat(sfName).st_size != 0:
            tNet = subnet.load(sfName)
            networks[stp] = tNet 
        else:
            subtypes.remove(stp)

    #print(len(networks),'\n',subtypes)
    #print(networks['A1'].biases)
    #sys.exit()
    # query file name
    qName = args.query #'hiv_data/hiv_refs_prrt_trim.fasta'


    # get the number of reference subtypes
    nSubtypes = len(subtypes)
    #print(nSubtypes)
    #sys.exit()

    # get the index of 'A1'; this marks the start index of PURE subtypes
    pIndex = subtypes.index('A1')
    #print(pIndex)
    #sys.exit()       
    
    ## read in the query sequences
    try:
        qSeqs = list(SeqIO.parse(args.query,'fasta'))
    except FileNotFoundError as e:
        eMsg = '\nThe query sequence file <{}> could not be found.'.format(args.query)
        eMsg += ' Please try again with correct sequence file name.\n'
        print(eMsg)
        sys.exit()

    ## check if the sequences were read properly
    if len(qSeqs) == 0:
        msg = 'Query sequences were not read properly'
        msg += '\nPlease run again with valid FASTA sequence file with at least one sequence\n'
        sys.exit(msg)
        
    #random.shuffle(qSeqs)
        
    # Create a directory to stote the output of the ANN_subtyping run
    timeNow = time.strftime('%Y-%m-%d-%H%M%S')
    zName = 'ann.' + timeNow 
    #print(zName)
    try:
        os.mkdir(zName)
        #pass
        
    except OSError as e:
        sys.exit(e)
                  
    oName = zName + '/' + args.outFile

    #sys.exit() 
    with open(oName,'w') as oh:
        oh.write('Name\tSubtype\n')
    
    for query in qSeqs:
        tMsg = calculateLikelihood(query,subtypes,nSubtypes,networks,pIndex,args,zName)
        
        print('{}'.format(tMsg))
        with open(oName,'a') as oh:
            oh.write('{}\n'.format(tMsg)) 
        


