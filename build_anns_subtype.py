from Bio import SeqIO
import umsgpack
import os, sys
import numpy as np
import subNet2 as subnet
import random

newData = True
#sType = '35_AD'

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


dPos = {'A' : 0,
        'C' : 1,
        'G' : 2,
        'T' : 3
      }
      
# adds n Ns at the beginning of the sequence
def padseq(s,n):
    return('N'*n+s)
    
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
    
cSize = 100
inSize = cSize * 4


# read in all the reference sequences
seqs = list(SeqIO.parse('training_HIV-ANN.fasta','fasta'))

# get the names of the subtypes
unique_subtypes = set([x.id.split('.')[0] for x in seqs])


# create a separate sequence file for each of the subtypes
for sType in unique_subtypes:
    nseqs = [seq for seq in seqs if seq.id.split('.')[0] == sType]
    fName = 'refAnn/{}.ref.fas'.format(sType)
    SeqIO.write(nseqs,fName,'fasta')
    #print('subtype = {}\tnumSeq = {}'.format(sType,len(nseqs)))


    rName = 'refAnn/' + sType + '.ref.fas'
    refs = list(SeqIO.parse(rName,'fasta'))
    rLen = len(refs)

    if newData:
        # get random order of sequences; first 3 are train, next 2 for validation and the last for test
        choices = random.sample(range(rLen),rLen)

        train = [refs[choices[i]] for i in range((rLen-1))]
        valid = [refs[choices[i]] for i in range(rLen-1,rLen)]

        #test = refs[5]
    
        trName = 'refAnn/' + sType + '.train.fas'
        SeqIO.write(train,trName,'fasta')
        vName = 'refAnn/' + sType + '.valid.fas'
        SeqIO.write(valid,vName,'fasta')
        #SeqIO.write(test,'refTypes/A1.test.fas','fasta')

    else:
        # just read in existing train and valid file
        trName = 'refAnn/' + sType + '.train.fas'
        train = list(SeqIO.parse(trName,'fasta'))
    
        vName = 'refAnn/' + sType + '.valid.fas'
        valid = list(SeqIO.parse(vName,'fasta')) 
    
    # assign a list for training inputs and a list for training targets
    training_inputs = list()
    training_targets = list()

    for seq in train:
        sequence = padseq(str(seq.seq).upper(),cSize)
        #print(sequence)
        for i in range(cSize,len(sequence)):
            start = i - cSize 
            ctx = sequence[start:i]
            ch = sequence[i]
            training_inputs.append(np.reshape(seq2array(ctx),(inSize,1)))
            training_targets.append(np.reshape(seq2array(ch),(4,1)))
        
    # create the training data by making tuple list 
    training_data = zip(training_inputs,training_targets)

    # assign a list for validating inputs and a list for validating targets
    validating_inputs = list()
    validating_targets = list()

    for seq in valid:
        sequence = padseq(str(seq.seq).upper(),cSize)
        #print(sequence)
        for i in range(cSize,len(sequence)):
            start = i - cSize 
            ctx = sequence[start:i]
            ch = sequence[i]
            validating_inputs.append(np.reshape(seq2array(ctx),(inSize,1)))
            validating_targets.append(dPos.get(ch,-1))

    # create the validate data by making tuple list 
    validate_data = zip(validating_inputs,validating_targets)

    # create the neural network
    net = subnet.Network([inSize, 500, 4],cost=subnet.CrossEntropyCost)

    net.large_weight_initializer()

    # run the network with validate_data
    sName = 'refAnn/' + sType + '.param'
    ec,ea,tc,ta, ba = net.SGD(training_data, 200, 20, 3.0, evaluation_data=validate_data,
        monitor_evaluation_accuracy=True, monitor_training_cost=False,
        monitor_training_accuracy=True,early_stopping_n=20,sName=sName)
        
    #print(sorted(ta))
    print('----------')
    #print(ea)
    acc = (float(ba) / len(sequence) ) * 100
    print('subtype={}\taccuracy={}'.format(sType,acc)) 
    #break       
