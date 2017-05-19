from Bio import SeqIO
#import umsgpack
import os, sys
import numpy as np
import subNet2 as subnet
import random
import json

if len(sys.argv) < 2:
   sys.exit('USAGE: train_anns_subtype.py <subtype>')

subtype = sys.argv[1]

#newData = True
#selectType = 'F1'
refDir = 'param_500'
sep = '|'
field = 1

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
    

def createReference(seqs,sts,sep,field):
    '''Creates separate reference files for each subtype
    '''
    
    for sub in sts:
        nseqs = [seq for seq in seqs if seq.id.upper().split(sep)[field] == sub]
        fName = refDir + '/{}.ref.fas'.format(sub)
        SeqIO.write(nseqs,fName,'fasta')


def trainSubtype(sType,cSize,refDir,rat=5):
    '''
    Reads in the reference subtype file
    creates all the contexts of size cSize
    Separates train and valid set based on ratio (ex. 5 means divide by 5; 4 for training, 1 for validation)
    trains the model
    create the param file
    '''    
    
    # read in the reference sequence file
    rName = refDir + '/' + sType + '.ref.1.fas'
    refs = list(SeqIO.parse(rName,'fasta'))
    print('Now training subtype {} using {} reference sequences\n'.format(sType,len(refs)))
    
    # get the input size for NN
    inSize = cSize * 4
    
    # create the input dataset
    inputs = []
    targets = []
    
    random.shuffle(refs)
    # get all the input contexts and their targets in the lists
    for seq in refs:
        sequence = padseq(str(seq.seq).upper().replace('-',''),cSize)

        for i in range(cSize,len(sequence)):
            start = i - cSize
            ctx = sequence[start:i]
            ch = sequence[i]
            if ch not in 'ACGT':
                continue
            if not set(ctx).issubset('ATCG'):
                continue
            
            inputs.append(ctx)
            targets.append(ch)        
           
    dSize = len(inputs)       
    # find starting index of validation data
    sIndex = int(dSize * ((rat - 1) / rat))
    print('Training data = {}\tValidation data = {}\n'.format(dSize, dSize-sIndex))
    #return    
    # create the training data
    training_inputs = []
    training_targets = []
    
    for i in range(sIndex):
        training_inputs.append(np.reshape(seq2array(inputs[i]),(inSize,1)))
        training_targets.append(np.reshape(seq2array(targets[i]),(4,1)))
    
    training_data = zip(training_inputs,training_targets)

    # create the validation data by making tuple list
    validation_inputs = []
    validation_targets = []
    
    for i in range(sIndex,len(inputs)):
        validation_inputs.append(np.reshape(seq2array(inputs[i]),(inSize,1)))
        validation_targets.append(dPos.get(targets[i],-1))
    
 
    validation_data = zip(validation_inputs,validation_targets)

    
    # create the neural network
    net = subnet.Network([inSize,500,4],cost=subnet.CrossEntropyCost)
    
    net.large_weight_initializer()
    
    # run the network with training_data, prrt: 300,20,3.0
    sName = refDir + '/' + sType + '.param'
    ec,ea,tc,ta, ba = net.SGD(training_data, 300, 20, 1.0, evaluation_data=validation_data,
        monitor_evaluation_accuracy=True, monitor_training_cost=False,
        monitor_training_accuracy=False,early_stopping_n=20,sName=sName)
    


    #print(sorted(ta))
    print('----------')
    print(ba)
    acc = (float(ba) / len(validation_inputs) ) * 100
    print('subtype={}\taccuracy={}'.format(sType,acc)) 
    #break       
    
    fName = refDir + '/accuracy.txt'  
    with open(fName,'a') as fh:
        fh.write('subtype={}\taccuracy={}\n'.format(sType,acc))


#********************************
cSize = 8

rat = 5

#print('Starting training')
# read in all the reference sequences
seqs = list(SeqIO.parse('hiv.prrt.cleaned.fas','fasta'))

#print('Read the sequences')
# get the names of the subtypes
unique_subtypes = sorted(list(set([x.id.upper().split(sep)[field] for x in seqs])))
#print(unique_subtypes)
#sys.exit(1)

## Create separate reference files for each subtypes
#createReference(seqs,unique_subtypes,sep,field)

trainSubtype(subtype,cSize,refDir,rat)

'''
for subtype in unique_subtypes:
    pName = refDir + '/' + subtype + '.param'
    if os.path.isfile(pName) and os.stat(pName).st_size != 0:
        continue
    
    trainSubtype(subtype,cSize,refDir,rat)
'''
