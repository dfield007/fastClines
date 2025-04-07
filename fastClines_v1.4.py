#!/usr/bin/env python
# Essential modules
import re, sys, getopt, itertools, os, math
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from operator import truediv
#cwd=os.getcwd()
print ("                                  ")
print ("-------------------------------------------------------")
print ("-    fastClines - Genome wide cline approximation     -")
print ("-   For whole genome data (poolSeq)                   -")
print ("-    requires *.sync data file                        -")
print ("-      David L. Field 09/05/2018                      -")
print ("-      07/01/2023 (last update)                       -")
print ("-      david.field@mq.edu.au                          -")
print ("-------------------------------------------------------")
print ("                                  ")
# written and tested for Python 3+

#  to run, you need the following arguments: 
#  (1) a txt file with a table of scaffolds to analyse (note *.sync files have to be in same folder), 
#  (2) a txt file with population names and spatial data, 
#  (3) output file name,
#  (4) the min depth to include a site, 
#  (5) the max depth to include a site, 
#  (6) minimum copies for an allele call (i.e. > 1, no singletons)
#  (7) minimum number of populations (or pools) with minimum depth to include a site, 
#  (8) delta p (allele frequency difference) threshold to estimate cline parameters, 
#  (9) how many of outer pools to use for delta p calculation (integer), 
#  (10) keep all sites (polymorphic or clinal only) (1 = polymorphic, 0 = clinal only),
#  (11) poolSeq data (1 = Y,0 = N),

#  min depth: if running on individual based data, make minimum depth the size of the smallest number of haploid genomes (i.e. 4x2 = 8),
#  min copies: if running on individual based data, make minimum copies = 1 (i.e. allow for singletons),
#  input: scaffold input list must have line breaks saved as Unix (LF). Use text wrangler or similar text editor,
#  # in the sync files, populations must appear in the required geographic order along a transect/cline. 
# Note, not all populations in the sync file have to be used
# Geographic pop details file must list the numbers in consecutive order from the full set present in sync
# e.g. in our example sync file, 20 pops are present but only pops 15,16,17,18,19,20 are utilised, as defined in the population file

#  To run: Examples:"
#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSELtiny.txt popDetails_Chr6_RosEL.txt clines_tiny.txt 10 300 2 6 0.8 1 0 1"
#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSELtest.txt popDetails_Chr6_RosEL.txt clines_test.txt 10 300 2 6 0.8 1 0 1"
#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSELtestTiny.txt popDetails_Chr6_RosEL.txt clines_testTiny.txt 10 300 2 6 0.8 1 0 1"
#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSEL_testb.txt popDetails_Chr6_RosEL.txt clines_testb.txt 10 300 2 6 0.8 1 0 1"
#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSEL_testc.txt popDetails_Chr6_RosEL.txt clines_testc 10 300 2 6 0.8 1 0 1"

#  python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSEL.txt popDetails_Chr6_RosEL.txt clines 10 300 2 6 0.8 1 0 1"

print ("Importing data... ")
print ('Input file 1 (Genome data - scaffold/Chromosome file list):', sys.argv[1])
print ('Input file 2 (Population details including Spatial data):', sys.argv[2])
print ('Output file:', sys.argv[3])
print ('Min depth:', sys.argv[4])
print ('Max depth:', sys.argv[5])
print ('Min copies for an allele call:', sys.argv[6])
print ('Min pool number passed:', sys.argv[7])
print ('Delta p threshold (clines):', sys.argv[8])
print ('Number of outer pools either side for delta p calculation (clines):', sys.argv[9])
print ('Output all sites (1 = all polymorphic, 0 = clinal only):', sys.argv[10])
print ('Pooled data (1 = PoolSeq, 0 = individual WGS):', sys.argv[11])

minDepthSeqInfo = int(sys.argv[4])
maxDepthSeqInfo = int(sys.argv[5])
minAlleleCount = int(sys.argv[6])
minPoolNum = int(sys.argv[7])
deltaPthresh = float(sys.argv[8])
poolAFD = int(sys.argv[9])
allSites = int(sys.argv[10])
poolSeq = int(sys.argv[11])

# test area
# outExt = 'clines'
# minDepthSeqInfo = 15
# maxDepthSeqInfo = 300
# minAlleleCount = 2
# minPoolNum = 6
# deltaPthresh = 0.8
# poolAFD = 1
# allSites = 0
# poolSeq = 1
# 

# Pop gen functions

def adjustData(LineList2, Pool, minDepthSeqInfo, maxDepthSeqInfo, minAlleleCount):
    # depth adjustments
    # adjustData(LineList2,thisPop,minDepthSeqInfo, maxDepthSeqInfo, minAlleleCount)
    # Pool = 15
    LineData = {}
    thisPoolVector=(int(Pool)+2)
    alleles=str(LineList2[thisPoolVector])
    alleles=alleles.split(":")
    alleles=alleles[0:4]
    alleles=list(map(float,alleles))
    LineData['pop']= Pool
    LineData['count']= alleles
    LineData['depth']= sum(LineData['count'])
    if (LineData['depth']>0):
        LineData['alleleFreq']=[x / LineData['depth'] for x in LineData['count']]
    if (LineData['depth']==0):
        LineData['alleleFreq']=[0,0,0,0]
    seqCode=["A","T","C","G"]            
    counter = 0
    for val in LineData['count']:
        LineData['count'][counter]=int(val)
        counter=counter+1
    LineData['base_present'] = [1 if x>=minAlleleCount else 0 for x in LineData['count']]
    counter=0
    LineData['count_adj']=LineData['count']
    LineData['bases_s']=['','','','']
    for allele in LineData['base_present']:
        LineData['bases_s'][counter]=int(allele)*seqCode[counter]
        counter=counter+1
    counter=0    
    LineData['bases_t']=''.join(LineData['bases_s'])
    LineData['count_adj'] = [q*w for q,w in zip(LineData['count_adj'],LineData['base_present'])]
    #LineData['depth_adj']=float(sum(LineData['count_adj']))
    LineData['depth']=sum(LineData['count'])
    LineData['depth_adj']=sum(LineData['count_adj'])
    if (LineData['depth_adj']>0):
        LineData['alleleFreq_adj']=[x / LineData['depth_adj'] for x in LineData['count_adj']]
    if (LineData['depth_adj']==0):
        LineData['alleleFreq_adj']=[0,0,0,0]
    LineData['minDepth_pass'] = 0
    if (LineData['depth_adj']>=minDepthSeqInfo and LineData['depth_adj']<= maxDepthSeqInfo):
        LineData['minDepth_pass'] = 1
    LineData['poly'] = 0
    if (len(LineData['bases_t'])>1):
        LineData['poly']=1
    alleles=[]
    return LineData

def divergeStats(Set1, Set2, Set1Div, Set2Div, p_1, q_1, p_2, q_2, minC):
    # David Field 12.08.16 
    # Set1 = 
    # Set2 = 
    #Set1Div, Set2Div, p_1, q_1, p_2, q_2, minC
    AFD,FET,Pbr,PbrA,Pt,PtA,Pb_raw,Pb_raw_adj,Pb_f,Pb_fA,F_Pb,F_Pb_adj,F_Pbr,F_PbrA,BP,counter,counter2=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
    Px,Qy,Py,Qx=[0,0,0,0]
    if (Set1[0]['minDepth_pass']==1 and Set2[0]['minDepth_pass']==1):
        AFD = abs(p_1-p_2)
        FET = -9 # note, not working yet
        # FET = stats.fisher_exact([[p_counts[0], p_counts[2]], [q_counts[0], q_counts[2]]])
        BP = 1
        Pb_raw = (p_1*q_2)+(q_1*p_2)
        PbrA = ((Set1Div[0]['Pi_adj']+Set2Div[0]['Pi_adj'])/2)
        Pbr = ((Set1Div[0]['Pi']+Set2Div[0]['Pi'])/2)
        Pmean = (p_1+p_2)/2
        Qmean = (q_1+q_2)/2
        Pt = 2*(Pmean*Qmean)
        t_C_Sets = [q+w for q,w in zip(Set1[0]['count_adj'],Set2[0]['count_adj'])]
        Dp_t_C_Sets = sum(t_C_Sets)
        Dp_t_C_Sets = float(Dp_t_C_Sets)
        AVal_Total = ((Dp_t_C_Sets)-1)/((Dp_t_C_Sets)-(2*minC)+1)
        # or 
        dp1 = Set1[0]['depth_adj']
        dp2 = Set2[0]['depth_adj']
        dp_m = (dp1+dp2)/2
        Aval_TotalMean  = ((dp_m)-1)/((dp_m)-(2*minC)+1)
        PtA = Pt*Aval_TotalMean
        # Dxy adjusted
        Pb_raw_adj = Pb_raw*Aval_TotalMean
        num2 = (Pt-Pbr)
        den2 = (Pt)
        F_Pbr=0
        if (den2>0):
            F_Pbr = num2/den2        
        num3 = (PtA-PbrA)
        den3 = (PtA)
        F_PbrA=0
        if (den3>0):
            F_PbrA = num3/den3
        # Fst from Dxy raw
        num = (Pb_raw-Pbr)
        den = (Pb_raw+Pbr)
        F_Pb = 0
        if (den>0):
            F_Pb = num/den
        # Dxy from Fst & Fst adj
        num4 = (1+F_PbrA)
        den4 = (1-F_PbrA)
        num5 = (1+F_Pbr)
        den5 = (1-F_Pbr)
        if (den4>0):
            Pb_fA = PbrA*(num4/den4)
        if (den5>0):
            Pb_f = Pbr*(num5/den5)
        # Fst from Dxy adjusted
        num6 = (Pb_raw_adj-PbrA)
        den6 = (Pb_raw_adj+PbrA)
        F_Pb_adj = 0
        if (den6>0):
            F_Pb_adj = num6/den6
    PairwiseDivergence = {}
    PairwiseDivergence['AFD'] = AFD
    PairwiseDivergence['FET'] = FET
    PairwiseDivergence['Pi_bar'] = Pbr
    PairwiseDivergence['Pi_bar_adj'] = PbrA
    PairwiseDivergence['Pi_T'] = Pt
    PairwiseDivergence['Pi_T_adj'] = PtA
    PairwiseDivergence['d_xy_raw'] = Pb_raw
    PairwiseDivergence['d_xy_raw_adj'] = Pb_raw_adj
    PairwiseDivergence['d_xy_fromFst'] = Pb_f
    PairwiseDivergence['d_xy_fromFstAdj'] = Pb_fA
    PairwiseDivergence['Fst_fromPi'] = F_Pbr # from unadjusted Pi. Same values come out from Fst_fromDxy
    PairwiseDivergence['Fst_fromPiAdj'] = F_PbrA
    PairwiseDivergence['Fst_fromDxy'] = F_Pb # from Dxy raw 
    PairwiseDivergence['Fst_fromDxyAdj'] = F_Pb_adj # from Dxy raw adjusted (new). Should be same as Fst_fromPiAdj
    PairwiseDivergence['BothPassed'] = BP
    return PairwiseDivergence

def diverseStats(poolData,minAlleleCount):
    # David Field 12.08.16 
    P_diversity = {}
    Pi_raw = 0; Pi_adj = 0; AdjVal = 0
    if (poolData['minDepth_pass']==1):
        if (poolData['poly']==1):
            Vals = [x for x in poolData['alleleFreq_adj'] if x > 0]
            Pi_raw = 2*Vals[0]*Vals[1]
            AdjVal = (float(poolData['depth_adj'])-1)/(float(poolData['depth_adj'])-(2*minAlleleCount)+1)
            Pi_adj = Pi_raw*AdjVal
    P_diversity['Pi'] = Pi_raw
    P_diversity['Pi_adj'] = Pi_adj
    P_diversity['AdjVal'] = AdjVal
    return P_diversity

def alleleFreq(allPools,pops_Pool,pops_allAdjustments):
    highestCount = max(x for x in allPools['count_allPools'] if x > 0)
    lowestCount = min(x for x in allPools['count_allPools'])
    lowestCountNonZero = min(x for x in allPools['count_allPools'] if x > 0)
    if (highestCount!=lowestCountNonZero):
        #print "\nAlternative Allele found"
        lowestCount = lowestCountNonZero
        P_allele = [1 if x==highestCount else 0 for x in allPools['count_allPools']]
        Q_allele = [1 if x==lowestCount else 0 for x in allPools['count_allPools']]
        alleleFreqs = defaultdict(list)
        for thisPop in pops_Pool:
            p_1 = max([q*w for q,w in zip(pops_allAdjustments[thisPop][0]['alleleFreq_adj'],P_allele)])
            q_1 = max([q*w for q,w in zip(pops_allAdjustments[thisPop][0]['alleleFreq_adj'],Q_allele)])
            alleleFreqs[thisPop].append(p_1)
            alleleFreqs[thisPop].append(q_1)
    if (highestCount==lowestCountNonZero):
        alleleFreqs = defaultdict(list)
        for thisPop in pops_Pool:
            #print '\n this pop: ', thisPop
            #print '\n pops_allAdjustments: ', pops_allAdjustments
            #print '\n pops_allAdjustments [1]: ', pops_allAdjustments[thisPop]
            #print '\n allele freq adj - this pop: ', pops_allAdjustments[thisPop][0]['alleleFreq_adj']
            p_1 = 1
            q_1 = 0
            alleleFreqs[thisPop].append(p_1)
            alleleFreqs[thisPop].append(q_1)
            #print '\n alleleFreqs: ', alleleFreqs
        minAllele  = min(x for x in allPools['count_allPools'] if x > 0)
    return alleleFreqs

def clineCentre(p,deltaPthresh,pops_Pool,poolAFD,InFilePop2df):
    #InFilePop2df['Dist']
    # deltaPthresh
    # p = [0.104,0.962,0.957,0.929,0.794,0.97]
    # spatialData = InFilePop2df['Dist']
    #poolAFD = 2
    #delta_calc = InFilePop2df['delta_calc'].values.tolist()
    #delta_calc = [val for sublist in delta_calc for val in sublist]
    #indices = [i for i, x in enumerate(delta_calc) if x == "1"]
    #deltaP = abs(p[0]-p[len(p)-1])
    deltaP = abs(np.average(p[0:poolAFD])-np.average(p[(len(p)-(poolAFD)):len(p)]))
    if (deltaP <= deltaPthresh):
        return -9
    if (deltaP > deltaPthresh):
        # flip alleles so p0 close to zero and p1 close to 1
        # this now done outside this function
        p_adj = p
        #if (p[0]>0.6):
        #    p_adj=[1-x for x in p]
        p0 = p_adj[0]
        p1 = p_adj[len(p_adj)-1]
        numerCentre = np.array(p_adj)-np.repeat(p0,len(p_adj))
        denomCentre = abs(np.array([p1-p0]*len(p_adj)))
        d_i = np.array(InFilePop2df['Dist']) # deme spans
        d_i = list(map(int, d_i))
        centre_output = sum(d_i) - sum((numerCentre/denomCentre) * d_i)
        return centre_output 

def clineWidth(p,deltaPthresh,pops_Pool,poolAFD,InFilePop2df):
    # p=[0.108695652,0.8,0.9,0.904761905,0.89,0.951891892]
    # p=[0.104,0.962,0.957,0.929,0.794,0.97]
    # clineWidth(p,deltaPthresh,pops_Pool,InFilePop2df)
    #deltaP = abs(p[0]-p[len(p)-1])
    deltaP = abs(np.average(p[0:poolAFD])-np.average(p[(len(p)-(poolAFD)):len(p)]))
    if (deltaP <= deltaPthresh):
        return -9
    if (deltaP > deltaPthresh):
        # flip alleles so p0 close to zero and p1 close to 1
        # this now done outside this function
        p_adj = p
        #if (p[0]>0.6):
        #    p_adj=[1-x for x in p]
        p0 = p_adj[0]
        p1 = p_adj[len(p_adj)-1]
        numerWidth = (np.array(p_adj)-np.repeat(p0,len(p_adj)))*(np.repeat(p1,len(p_adj))-np.array(p_adj))
        denomWidth = np.array([p1-p0]*len(p_adj))**2
        d_i = np.array(InFilePop2df['Dist'])
        d_i = list(map(int, d_i))
        width_output = 4 * sum((numerWidth/denomWidth) * d_i)
        return width_output

# random functions
def MyDivision(num, denom):
    if denom==0:
        return "NaN"
    if (denom=='NaN' or denom=='NaN'):
        return "NaN"
    else:
        return (num/denom)
def MySubtraction(first,second):
    if (first=='NaN' or second=='NaN'):
        return 'NaN'
    if (first!='NaN' and second!='NaN'):
        return (first-second)
def MyMultiplication(first, second):
    if second=='NaN':
        return 'NaN'
    if second!='NaN':
        return (first * second)
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def round_down(num, divisor):
    # test
    return (num - (num%divisor))    

# InfilePop will become the table including the subsets of populations to use for cline calculations

# InFilePopTemp = 'popDetails_Chr6_RosEL.txt'
# sys argument version
InFilePop = open(sys.argv[2], 'r')
# InFilePop = open('popDetails_Chr6_RosEL.txt', 'r')

InFilePop2 = [i.split()[0:] for i in InFilePop.readlines()]
InFilePop.close() 
numPops = len(InFilePop2)-1
# pandas data frame to interpret pop IDs
pops = list(range(1, numPops+1))
InFilePop2df = pd.DataFrame(InFilePop2[1:numPops+1])
InFilePop2df.columns = [InFilePop2[0:1][0]]
print('')
print(InFilePop2df)

pops_ID = InFilePop2df['ID'].values.tolist()
pops_ID = [val for sublist in pops_ID for val in sublist]

pops_Pool = InFilePop2df['Pool'].values.tolist()
pops_Pool = [val for sublist in pops_Pool for val in sublist]

comboPops_nums = []
comboPops_ID = []
comboPops_Pool = []

#print('\n population nums: ', pops)
#print('\n population IDs: ', pops_ID)
#print('\n population Pools in genome file: ', pops_Pool)

for subset in itertools.combinations(pops, 2):
    comboPops_nums.append(subset) 
#print('\n population pairs: ' , comboPops_nums)

for subset in itertools.combinations(pops_ID, 2):
    comboPops_ID.append(subset) 
#print('\n population pairs IDs: ' , comboPops_ID)

for subset in itertools.combinations(pops_Pool, 2):
    comboPops_Pool.append(subset) 
#print('\n population pairs Pools: ' , comboPops_Pool)

comboPops_header_sites = []
for subset in itertools.combinations(pops, 2):
    comboPops_header_sites.append(str(subset[0]) + '_' + str(subset[1]))

# two new alternatives for 
# pop ID     
comboPopsID_header_sites = []
for subset in itertools.combinations(pops_ID, 2):
     comboPopsID_header_sites.append(str(subset[0]) + '_' + str(subset[1]))
# pool number in sync file     
comboPopsPool_header_sites = []
for subset in itertools.combinations(pops_Pool, 2):
    comboPopsPool_header_sites.append(str(subset[0]) + '_' + str(subset[1]))

# # header site output - Original
# header_sites_start = ["scaffold", "position", "LG", "cM", "ref", "bases_t", "allele_num", "1or2bases", "minDepth_pools", "minDepth_pass"]

# headers_sites_pops = ['p_{0}'.format(i) for i in pops_Pool] + ['poly_pool_{0}'.format(i) for i in pops_Pool] \
# + ['dpthAdj_{0}'.format(i) for i in pops_Pool] + ['dpthPass_{0}'.format(i) for i in pops_Pool] + ['pi_{0}'.format(i) for i in pops_Pool] \
# + ['piAdj_{0}'.format(i) for i in pops_Pool]

# headers_sites_clines = ["centre", "width"]

# # two new headers AFD= Allele Frequency Difference, FET=Fischer Exact Test.
# headers_sites_pairs = ['dpthPolyPass_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['AFD_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['FET_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['piBar_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['piBarAdj_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['piT_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['piTadj_{0}'.format(i) for i in comboPopsPool_header_sites]\
# + ['dXYraw_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['dXYfromFstadj_{0}'.format(i) for i in comboPopsPool_header_sites]\
# + ['FstPi_{0}'.format(i) for i in comboPopsPool_header_sites] \
# + ['FstPiAdj_{0}'.format(i) for i in comboPopsPool_header_sites]\
# + ['FstfromDxy_{0}'.format(i) for i in comboPopsPool_header_sites]

#
# header site output - NEW streamlined approach 2023

# if poolSeq = 1 ; reports Adjusted values (for bionmial sampling), if poolSeq = 0 ; reports outputs from unadjusted values

header_sites_start = ["scaffold", "position", "LG", "cM", "ref", "bases_t", "allele_num"] 
# 7 columns (3 deleted). Deleted: "1or2bases", "minDepth_pools", "minDepth_pass"

headers_sites_pops = ['p_{0}'.format(i) for i in pops_Pool] + ['dpthAdj_{0}'.format(i) for i in pops_Pool] \
+ ['pi_{0}'.format(i) for i in pops_Pool] 
# + ['piAdj_{0}'.format(i) for i in pops_Pool] 
# poly_pool & depthPass deleted. piAdj deleted, 
# but the value in each will depend on poolSeq input (i.e. 1 = adjusted, 0 = nonadjusted)

headers_sites_clines = ["centre", "width"]

headers_sites_pairs = ['AFD_{0}'.format(i) for i in comboPopsPool_header_sites] \
+ ['piBar_{0}'.format(i) for i in comboPopsPool_header_sites] \
+ ['piT_{0}'.format(i) for i in comboPopsPool_header_sites] \
+ ['dXYraw_{0}'.format(i) for i in comboPopsPool_header_sites] \
+ ['dXYfromFst_{0}'.format(i) for i in comboPopsPool_header_sites]\
+ ['FstfromPi_{0}'.format(i) for i in comboPopsPool_header_sites] \
+ ['FstfromDxy_{0}'.format(i) for i in comboPopsPool_header_sites]

# dpthPolyPass, FET deleted. All adj columns deleted,
# but the value in each will depend on poolSeq input (i.e. 1 = adjusted, 0 = nonadjusted)

OutputStringHeaderTempFile = '\t'.join(header_sites_start + headers_sites_pops + headers_sites_clines + headers_sites_pairs)
#228 columns in total previously for 6 pops
# len(header_sites_start+headers_sites_pops+headers_sites_clines+headers_sites_pairs) 
#len(header_sites_start) # 7
#len(headers_sites_pops) # 3 stats x 6 pops = 18
#len(headers_sites_clines) # centre, width = 2 
#len(headers_sites_pairs) # 7 stats x 15 pairwise = 105
# for this many pops = 132 columns

# header depth output
header_sites_dpth_start = ["scaffold", "position", "LG", "cM", "ref", "bases_t", "allele_num", "1or2bases", "minDepth_pools", "minDepth_pass"]
headers_sites_dpth_pops = ['p_{0}'.format(i) for i in pops_Pool] + ['dpthAdj_{0}'.format(i) for i in pops_Pool] \
+ ['dpthPass_{0}'.format(i) for i in pops_Pool]
headers_sites_dpth_pairs = ['dpthPolyPass_{0}'.format(i) for i in comboPopsPool_header_sites]
OutputStringHeaderTempFile_dpth = '\t'.join(header_sites_dpth_start + headers_sites_dpth_pops + headers_sites_dpth_pairs)

# tracking: Original
headerCombined = header_sites_start + headers_sites_pops + headers_sites_clines + headers_sites_pairs
headerCombined_dpth = header_sites_dpth_start + headers_sites_dpth_pops + headers_sites_dpth_pairs
# indexing statistic locations
# site output file
# 6 single pop stats
statsOrderSingle = ['p','poly_pool','dpthAdj','dpthPass','pi','piAdj']
# 2 cline stats
statsOrderClines = ['centre','width',]

# 12 pairwise stats 
statsOrderPairs = ['dpthPolyPass','AFD','FET','piBar','piBarAdj','piT','piTadj','dXYraw','dXYfromFstadj','FstPi','FstPiAdj','FstfromDxy']

# dpth output file (3 single pop, 1 pairwise)
statsOrderSingle_dpth = ['p','dpthAdj', 'dpthPass']; statsOrderPairs_dpth = ['dpthPolyPass']

# tracking: new version

# tracking: new
headerCombined = header_sites_start + headers_sites_pops + headers_sites_clines + headers_sites_pairs
headerCombined_dpth = header_sites_dpth_start + headers_sites_dpth_pops + headers_sites_dpth_pairs
# indexing statistic locations
# site output file
# 3 single pop stats
statsOrderSingle = ['p','dpthAdj','pi']
# 2 cline stats
statsOrderClines = ['centre','width',]

# 7 pairwise stats 
statsOrderPairs = ['AFD','piBar','piT','dXYraw','dXYfromFstadj','FstPi','FstfromDxy']

# dpth output file (3 single pop, 1 pairwise)
#statsOrderSingle_dpth = ['p','dpthAdj', 'dpthPass']; statsOrderPairs_dpth = ['dpthPolyPass']

# single pop stats
# initial vals
statsColumnList = defaultdict(list)
thisStart = len(header_sites_start)
thisEnd = (thisStart+numPops)

singlePopStatsCounter = 1
counter = 1
for thisStat in statsOrderSingle: 
    # thisStat = 'p'
    # thisStat = 'poly_pool'
    if counter == 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnList[thisStat].append(thisStart)
        statsColumnList[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+numPops
    if counter != 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnList[thisStat].append(thisStart)
        statsColumnList[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+numPops
    counter = counter+1

# clines
# initial vals
statsColumnListClines = defaultdict(list)
thisEnd = (thisStart+1)
counter = 1
for thisStat in statsOrderClines: 
    # thisStat = 'centre'
    # thisStat = 'width'
    if counter == 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnListClines[thisStat].append(thisStart)
        statsColumnListClines[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+1
    if counter != 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnListClines[thisStat].append(thisStart)
        statsColumnListClines[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+1
    counter = counter+1

# pop pairs
# initial vals
statsColumnListPairs = defaultdict(list)
numPairs = len(comboPopsPool_header_sites)
thisStart = thisEnd-1
thisEnd = (thisStart+numPairs)
counter = 1
for thisStat in statsOrderPairs: 
    # thisStat = 'dpthPolyPass'
    if counter == 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnListPairs[thisStat].append(thisStart)
        statsColumnListPairs[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+numPairs
    if counter != 1:
        thisRange = headerCombined[thisStart:thisEnd]
        statsColumnListPairs[thisStat].append(thisStart)
        statsColumnListPairs[thisStat].append(thisEnd)
        thisStart = thisEnd
        thisEnd = thisStart+numPairs
    counter = counter+1

# depth file
# statsColumnList_dpth = defaultdict(list)
# thisStart = len(header_sites_dpth_start)
# thisEnd = (thisStart+numPops)
# singlePopStatsCounter = 1
# counter = 1
# for thisStat in statsOrderSingle_dpth: 
#     if counter == 1:
#         thisRange = headerCombined_dpth[thisStart:thisEnd]
#         statsColumnList_dpth[thisStat].append(thisStart)
#         statsColumnList_dpth[thisStat].append(thisEnd)
#         thisStart = thisEnd
#         thisEnd = thisStart+numPops
#     if counter != 1:
#         thisRange = headerCombined_dpth[thisStart:thisEnd]
#         statsColumnList_dpth[thisStat].append(thisStart)
#         statsColumnList_dpth[thisStat].append(thisEnd)
#         thisStart = thisEnd
#         thisEnd = thisStart+numPops
#     counter = counter+1
            
# Depth pop pairs
# initial vals
# statsColumnList_Pairs_dpth = defaultdict(list)
# numPairs = len(comboPops_nums)
# thisStart = thisEnd-numPops
# thisEnd = (thisStart+numPairs)
# counter = 1
# for thisStat in statsOrderPairs_dpth: 
#     if counter == 1:
#         thisRange = headerCombined_dpth[thisStart:thisEnd]
#         statsColumnList_Pairs_dpth[thisStat].append(thisStart)
#         statsColumnList_Pairs_dpth[thisStat].append(thisEnd)
#         thisStart = thisEnd
#         thisEnd = thisStart+numPairs
#     if counter != 1:
#         thisRange = headerCombined_dpth[thisStart:thisEnd]
#         statsColumnList_Pairs_dpth[thisStat].append(thisStart)
#         statsColumnList_Pairs_dpth[thisStat].append(thisEnd)
#         thisStart = thisEnd
#         thisEnd = thisStart+numPairs    
#     counter = counter+1

# read 
InFileTest = open(sys.argv[1], 'r')
# InFileTest = open('scaffold_Chr6_v3_5_ROSEL.txt', 'r')
# InFileTest = open('scaffold_Chr6_v3_5_ROSELtest.txt', 'r')
# InFileTest = open('scaffold_Chr6_v3_5_ROSELtestTiny.txt', 'r')
# InFileTest = open('scaffold_Chr6_v3_5_ROSEL_testb.txt', 'r')

InFile  = [i.split()[0:] for i in InFileTest.readlines()]
InFileTest.close() 
lengthInFile=len(InFile)
numLines = float(sum(1 for line in InFile))

# Initialise 
LineNumber = 0; thisScaff = []; thisPos = []; theDataSummary = []; theDataFull = []

# Scaffold list
for Line in InFile:
    # Line = InFile[1]
    thisScaff = []; thiscM = []; thisLG = []; theDataTemp = []
    #PercentCompleteWindows = 0
    if LineNumber > 0:
        thisScaff = str(Line[2])
        thisLG = str(Line[0])
        thiscM = Line[3]
        print ('\n ')
        print("\n Running analyses... Scaffold/LG ",int(round(LineNumber,0)),"of",int(round((numLines-1),0)))
        #if windowOnly == 0:
        InFile2 = open(thisScaff + '.sync', 'r')
        InFile2lines  = [i.split()[0:] for i in InFile2.readlines()]
        InFile2.close() 
        InFile2lineNum = len(InFile2lines)
        totalScaffSize = InFile2lineNum
        print('\n thisScaffold name: ', thisScaff)
        print(' LG: ', thisLG)
        print(' Number of sites: ', InFile2lineNum)
        with open(thisScaff + '.sync') as InFile2:
            #OutFileTemp = open(thisScaff + '_OutputSites_' + sys.argv[3] + '.txt', 'w')
            OutFileTemp = open(thisScaff + '_OutputClines' + '.txt', 'w')
            #OutFileTemp = open(thisScaff + '_OutputSites_' + 'cline_testb' + '.txt', 'w')
            # OutFileTemp_dpth = open(thisScaff + '_OutputSites_dpth.txt', 'w')
            # collecting line and position indexes
            # depth file
            lineNum_dpth = 0
            lineNumCounter_dpth = []
            positionCounter_dpth = []
            lineNum_sites = 0
            lineNumCounter_sites = []
            positionCounter_sites = []
            # Initialise
            startFlag=0; stopcounter=0; switchOn=0; numNonBiallele=0; dataOutputFile=0; lineNumberScaffold=0
            increments_small = np.array([x / 1000.0 for x in range(10, 1000, 10)] )
            increments_lines_small = increments_small*np.repeat(InFile2lineNum,len(increments_small))
            increments_lines_small = [int(x) for x in increments_lines_small]
            increments_big = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
            increments_lines_big = increments_big*np.repeat(InFile2lineNum,len(increments_big))
            increments_lines_big = [int(x) for x in increments_lines_big]
            countProg = int(0)
            print ("\n Processing sites: ")
            OutFileTemp.write(OutputStringHeaderTempFile+"\n")
            #OutFileTemp_dpth.write(OutputStringHeaderTempFile_dpth+"\n")
            for line in InFile2:
                # line = InFile2[41]
                OutputString = 1
                # this line extract
                LineList2 = line.split()[0:]
                currentScaff = str(LineList2[0])
                currentPos = int(LineList2[1])
                #sys.stdout.write('%s' % (currentPos)+'%')
                #sys.stdout.flush()
                #print ('Position:', currentPos)
                posCounter = currentPos
                refAllele = str(LineList2[2])
                position_passed = []
                pops_allAdjustments = defaultdict(list)
                for thisPop in pops_Pool:
                    pops_allAdjustments[thisPop].append(adjustData(LineList2,thisPop,minDepthSeqInfo, maxDepthSeqInfo, minAlleleCount))
                allPools={}
                minDepthPassed = 0
                for thisPop in pops_Pool:
                    minDepthPassed = float(pops_allAdjustments[thisPop][0]['minDepth_pass']) + minDepthPassed
                #print ('pops_allAdjustments', pops_allAdjustments)
                allPools['minDepth_pools'] = minDepthPassed
                allPools['minDepth_thresh'] = 0
                if (allPools['minDepth_pools'] >= minPoolNum):
                    allPools['minDepth_thresh'] = 1
                # Check if locus has >2 bases with a different set of most common alleles present in pools
                totalCounts = [0]*4            
                for thisPop in pops_Pool:
                    totalCounts = [q+w for q,w in zip(totalCounts,pops_allAdjustments[thisPop][0]['count_adj'])]
                allPools['count_allPools'] = totalCounts
                allPools['base_present'] = [1 if x>=minAlleleCount else 0 for x in allPools['count_allPools']]
                # Alleles present
                counter = 0
                seqCode = ["A","T","C","G"]
                allPools['bases_s'] = ['','','','']
                for allele in allPools['base_present']:
                    allPools['bases_s'][counter] = int(allele)*seqCode[counter]
                    counter = counter+1
                counter = 0    
                allPools['bases_t'] = ''.join(allPools['bases_s'])
                allPools['numAlleles'] = len(allPools['bases_t'])
                allPools['OneOrTwoBp'] = 0
                if (allPools['minDepth_thresh'] == 0):
                    # skips output if below threshold depth across pools
                    next
                if (sum(allPools['base_present'])<=2 and allPools['minDepth_thresh'] != 0 and allPools['numAlleles'] <= 2):
                    # new approach
                    p=[]; dpthAdj=[]; dpthPass=[]; pi=[]; 
                    Cline_Centre=[]; Cline_Width=[]; AFD=[]; 
                    pi_bar=[]; pi_T=[]; d_xy_raw=[]; d_xy_fromFst=[]; d_xy_fromFstAdj=[] 
                    Fst_fromPi=[]; Fst_fromDxy=[]
                    allPools['OneOrTwoBp'] = 1
                    # Diversity statistics 
                    pops_diversityStats = defaultdict(list)
                    for thisPop in pops_Pool:
                        pops_diversityStats[thisPop].append(diverseStats(pops_allAdjustments[thisPop][0],minAlleleCount))
                    # Allele frequencies
                    alleleFreq_allPops = alleleFreq(allPools,pops_Pool,pops_allAdjustments)
                    #print ('alleleFreq_allPops:', alleleFreq_allPops)
                    for thisPop in pops_Pool:
                        p.append(alleleFreq_allPops[thisPop][0])
                        dpthAdj.append(pops_allAdjustments[thisPop][0]['depth_adj'])
                        dpthPass.append(pops_allAdjustments[thisPop][0]['minDepth_pass'])
                    #print ('p:', p)
                    # flip alleles so p0 close to zero and p1 close to 1
                    p_flip = p
                    if (p[0]>0.6 and p[len(p)-1]<0.6):
                        p_flip=[1-x for x in p]
                    #print ('p flip:', p_flip)
                    # p_flip not in the append stats below
                    # Pairwise divergence statistics
                    pairs_divergenceStats = defaultdict(list)
                    for thisPair in comboPops_Pool:
                        pairs_divergenceStats[thisPair].append(divergeStats(pops_allAdjustments[thisPair[0]],pops_allAdjustments[thisPair[1]], \
                        pops_diversityStats[thisPair[0]],pops_diversityStats[thisPair[1]], \
                        alleleFreq_allPops[thisPair[0]][0],alleleFreq_allPops[thisPair[0]][1], \
                        alleleFreq_allPops[thisPair[1]][0],alleleFreq_allPops[thisPair[1]][1],minAlleleCount))
                    #for thisPair in comboPops_Pool:
                    #    dpthPolyPass.append(pairs_divergenceStats[thisPair][0]['BothPassed'])
                    # Cline estimates
                    centre = clineCentre(p_flip,deltaPthresh,pops_Pool,poolAFD,InFilePop2df)
                    width = clineWidth(p_flip,deltaPthresh,pops_Pool,poolAFD,InFilePop2df)
                    # Depth file write - blocked in new version
                    #OutputStringStart_dpth = [thisScaff,currentPos,thisLG,thiscM,refAllele,allPools['bases_t'],allPools['numAlleles'],allPools['OneOrTwoBp'],allPools['minDepth_pools'],allPools['minDepth_thresh']]
                    #lineNum_dpth = lineNum_dpth + 1
                    # if OutputString==1 and lineNum_dpth!=0:
                    #     OutputStringData_dpth = OutputStringStart_dpth + p + dpthAdj + dpthPass + dpthPolyPass
                    #     OutFileTemp_dpth.write("\t".join(map(str, OutputStringData_dpth))+"\n")
                    #     lineNumCounter_dpth.append(float(lineNum_dpth))
                    #     positionCounter_dpth.append(float(currentPos))    
                # Records all polymorphic sites (clinal or non-clinal) (if allSites==1)
                if (sum(allPools['base_present'])<=2 and allPools['minDepth_thresh'] != 0 and allPools['numAlleles'] == 2 and allSites == 1):
                    outputStringStart = [thisScaff,currentPos,thisLG,thiscM,refAllele,allPools['bases_t'],allPools['numAlleles']]
                    poly_pool=[]; dpthAdj=[]; dpthPass=[]; pi=[]; piAdj=[]
                    Cline_Centre=[]; Cline_Width=[]
                    dpthPolyPass=[]; AFD=[]; FET=[]
                    pi_bar=[]; pi_bar_adj=[]; pi_T=[]; pi_T_adj=[]
                    d_xy_raw=[];d_xy_raw_adj=[]; d_xy_fromFst=[]; d_xy_fromFstAdj=[]
                    Fst_fromPi=[]; Fst_fromPiAdj=[]; Fst_fromDxy=[]; Fst_fromDxyAdj=[]
                    # Append outputs
                    Cline_Centre.append(centre)
                    Cline_Width.append(width)
                    for thisPop in pops_Pool:
                        poly_pool.append(pops_allAdjustments[thisPop][0]['poly'])
                        dpthAdj.append(pops_allAdjustments[thisPop][0]['depth_adj'])
                        dpthPass.append(pops_allAdjustments[thisPop][0]['minDepth_pass'])
                        pi.append(pops_diversityStats[thisPop][0]['Pi'])
                        piAdj.append(pops_diversityStats[thisPop][0]['Pi_adj'])
                    for thisPair in comboPops_Pool:
                        dpthPolyPass.append(pairs_divergenceStats[thisPair][0]['BothPassed'])
                        AFD.append(pairs_divergenceStats[thisPair][0]['AFD'])
                        FET.append(pairs_divergenceStats[thisPair][0]['FET'])
                        pi_bar.append(pairs_divergenceStats[thisPair][0]['Pi_bar'])
                        pi_bar_adj.append(pairs_divergenceStats[thisPair][0]['Pi_bar_adj'])
                        pi_T.append(pairs_divergenceStats[thisPair][0]['Pi_T'])
                        pi_T_adj.append(pairs_divergenceStats[thisPair][0]['Pi_T_adj'])
                        d_xy_raw.append(pairs_divergenceStats[thisPair][0]['d_xy_raw'])
                        d_xy_raw_adj.append(pairs_divergenceStats[thisPair][0]['d_xy_raw_adj'])
                        d_xy_fromFst.append(pairs_divergenceStats[thisPair][0]['d_xy_fromFst'])
                        d_xy_fromFstAdj.append(pairs_divergenceStats[thisPair][0]['d_xy_fromFstAdj'])
                        Fst_fromPi.append(pairs_divergenceStats[thisPair][0]['Fst_fromPi'])
                        Fst_fromPiAdj.append(pairs_divergenceStats[thisPair][0]['Fst_fromPiAdj'])
                        Fst_fromDxy.append(pairs_divergenceStats[thisPair][0]['Fst_fromDxy'])
                        Fst_fromDxyAdj.append(pairs_divergenceStats[thisPair][0]['Fst_fromDxyAdj'])
                    # Main site output write
                    lineNum_sites = lineNum_sites + 1
                    if poolSeq == 1 and OutputString==1 and lineNum_sites!=0:
                        # Note _Adj calulcations used (where applicable) if poolSeq = 1
                        OutputStringData = outputStringStart + p_flip + dpthAdj + piAdj + Cline_Centre + Cline_Width \
                        + AFD + pi_bar_adj + pi_T_adj + d_xy_raw_adj + d_xy_fromFstAdj + Fst_fromPiAdj + Fst_fromDxyAdj
                        OutFileTemp.write("\t".join(map(str, OutputStringData))+"\n")
                        lineNumCounter_sites.append(float(lineNum_sites))
                        positionCounter_sites.append(float(currentPos))
                    if poolSeq == 0 and OutputString==1 and lineNum_sites!=0:
                        OutputStringData = outputStringStart + p_flip + dpthAdj + pi + Cline_Centre + Cline_Width \
                        + AFD + pi_bar + pi_T + d_xy_raw + d_xy_fromFst + Fst_fromPi + Fst_fromDxy
                        # len(OutputStringData)
                        OutFileTemp.write("\t".join(map(str, OutputStringData))+"\n")
                        lineNumCounter_sites.append(float(lineNum_sites))
                        positionCounter_sites.append(float(currentPos))
                # Records only sites with clines (if allSites==0). 
                if (sum(allPools['base_present'])<=2 and allPools['minDepth_thresh'] != 0 and allPools['numAlleles'] <= 2 and allSites == 0 and centre !=-9):
                    outputStringStart = [thisScaff,currentPos,thisLG,thiscM,refAllele,allPools['bases_t'],allPools['numAlleles']]
                    poly_pool=[]; dpthAdj=[]; dpthPass=[]; pi=[]; piAdj=[]
                    Cline_Centre=[]; Cline_Width=[]
                    dpthPolyPass=[]; AFD=[]; FET=[]
                    pi_bar=[]; pi_bar_adj=[]; pi_T=[]; pi_T_adj=[]
                    d_xy_raw=[];d_xy_raw_adj=[]; d_xy_fromFst=[]; d_xy_fromFstAdj=[]
                    Fst_fromPi=[]; Fst_fromPiAdj=[]; Fst_fromDxy=[]; Fst_fromDxyAdj=[]
                    # Append outputs
                    Cline_Centre.append(centre)
                    Cline_Width.append(width)
                    for thisPop in pops_Pool:
                        poly_pool.append(pops_allAdjustments[thisPop][0]['poly'])
                        dpthAdj.append(pops_allAdjustments[thisPop][0]['depth_adj'])
                        dpthPass.append(pops_allAdjustments[thisPop][0]['minDepth_pass'])
                        pi.append(pops_diversityStats[thisPop][0]['Pi'])
                        piAdj.append(pops_diversityStats[thisPop][0]['Pi_adj'])
                    for thisPair in comboPops_Pool:
                        dpthPolyPass.append(pairs_divergenceStats[thisPair][0]['BothPassed'])
                        AFD.append(pairs_divergenceStats[thisPair][0]['AFD'])
                        FET.append(pairs_divergenceStats[thisPair][0]['FET'])
                        pi_bar.append(pairs_divergenceStats[thisPair][0]['Pi_bar'])
                        pi_bar_adj.append(pairs_divergenceStats[thisPair][0]['Pi_bar_adj'])
                        pi_T.append(pairs_divergenceStats[thisPair][0]['Pi_T'])
                        pi_T_adj.append(pairs_divergenceStats[thisPair][0]['Pi_T_adj'])
                        d_xy_raw.append(pairs_divergenceStats[thisPair][0]['d_xy_raw'])
                        d_xy_raw_adj.append(pairs_divergenceStats[thisPair][0]['d_xy_raw_adj'])
                        d_xy_fromFst.append(pairs_divergenceStats[thisPair][0]['d_xy_fromFst'])
                        d_xy_fromFstAdj.append(pairs_divergenceStats[thisPair][0]['d_xy_fromFstAdj'])
                        Fst_fromPi.append(pairs_divergenceStats[thisPair][0]['Fst_fromPi'])
                        Fst_fromPiAdj.append(pairs_divergenceStats[thisPair][0]['Fst_fromPiAdj'])
                        Fst_fromDxy.append(pairs_divergenceStats[thisPair][0]['Fst_fromDxy'])
                        Fst_fromDxyAdj.append(pairs_divergenceStats[thisPair][0]['Fst_fromDxyAdj'])
                    # Main site output write
                    lineNum_sites = lineNum_sites + 1                    
                    if poolSeq == 1 and OutputString==1 and lineNum_sites!=0:
                        # Note _Adj calulcations used (where applicable) if poolSeq = 1
                        OutputStringData = outputStringStart + p_flip + dpthAdj + piAdj + Cline_Centre + Cline_Width \
                        + AFD + pi_bar_adj + pi_T_adj + d_xy_raw_adj + d_xy_fromFstAdj + Fst_fromPiAdj + Fst_fromDxyAdj
                        OutFileTemp.write("\t".join(map(str, OutputStringData))+"\n")
                        lineNumCounter_sites.append(float(lineNum_sites))
                        positionCounter_sites.append(float(currentPos))
                    if poolSeq == 0 and OutputString==1 and lineNum_sites!=0:
                        OutputStringData = outputStringStart + p_flip + dpthAdj + pi + Cline_Centre + Cline_Width \
                        + AFD + pi_bar + pi_T + d_xy_raw + d_xy_fromFst + Fst_fromPi + Fst_fromDxy
                        # len(OutputStringData)
                        OutFileTemp.write("\t".join(map(str, OutputStringData))+"\n")
                        lineNumCounter_sites.append(float(lineNum_sites))
                        positionCounter_sites.append(float(currentPos))
                lineNumberScaffold = lineNumberScaffold+1
                if lineNumberScaffold in increments_lines_small:
                    sys.stdout.write('%s' % ('.'))
                    sys.stdout.flush()
                if lineNumberScaffold in increments_lines_big:
                    increments_big[9]*100
                    sys.stdout.write('%s' % (increments_big[countProg]*100)+'%')
                    sys.stdout.flush()
                    countProg = countProg+1
                    InFile2.flush()
        OutFileTemp.close()
        #OutFileTemp_dpth.close()
    LineNumber = LineNumber + 1
    #LineNum_sync = LineNum_sync + 1
print ("\n ")
print ("*** Analysis complete ***")
print (" ")