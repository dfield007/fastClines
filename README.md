# fastClines
This is a Python package for performing rapid approximation of geographic cline parameters for SNP genetic data as outlined in Field et al., 'Genome-wide cline analysis identifies new locus contributing to a barrier to gene flow across an _Antirrhinum_ hybrid zone' (in review) https://www.biorxiv.org/content/10.1101/2025.02.17.638607v1.full. 
The input data consists of genome wide SNP data and the geographic location of populations. The main output consists of cline width and cline centre for each SNP locus.

## Table of contents
1. Overview
2. Installation
3. Using fastClines
4. Citing fastClines
5. Authors and license information

# Overview
This script calculates geographic cline parameters for spatially explicit genomic data across a hybrid zone or any sharp transition in space. This includes (i) cline width, defined as the inverse of the maximum gradient in allele frequencies (p) and (ii) cline centre, defined as the centre of mass of allele frequencies (p) where p ~ 0.5. This requires that genetic data is available for multiple individuals grouped (or pools) from a set of demes (sub-populations) along a (roughly) one-dimensional transect or 2-D that can be collapsed to 1-D. FastClines is an efficient method to quickly calculate proxies of cline properties for large numbers of loci typical of genomic scale data.

This method is based on the premise that cline width is proportional to cummulative heterozygosity. For steeper clines, this accumulates in a smaller geographic area. For shallower clines, cummulative heterozygosity is more spread out in space. Cline centre is approximated via centre of mass of allele frequencies. The method can handle SNP loci which are non-diagnostic between parental species (i.e. polymorphic) and uneven geographic spacing of demes along the transect. Parental allele frequencies on either end of the cline are calculated from minimum and maximum allele frequencies on either ends of the transect. 

The fastClines package also calculate various other site based population genetic metrics including Allele Frequency difference between the outer most population pairs (presumed to be parental reference populations) nucleotide diveristy within (pi_w), absolute nucleotide divergence between populations (Dxy), relative differentiation (Fst). Some of these metrics are calculated in a few different ways (described below), but are not core functions of the package.

The fastClines approach can only handle biallelic diploid SNP data for sequence data in sync file format used for pooled whole genome sequence data. To aid use of the package, we provide a script to convert vcf files to sync format (vcf_to_sync.py). 

Running the method requires:

1. Genomic data (.sync format) - organised into a series of populations (i.e. demes) along a geographic transect. 
2. Geographic information (.txt format) - listing information about the names of the populations, the geographic distances between populations and their specific order along the transect (see below)
3. Selecting user specified filtering values and other arguments (see using FastClines).

# Installation

Simply place the python script file (fastClines_v1.4.py) into your working directory of choice from the github version. 

## Dependencies

FastClines is built with Python 3.0 + using Numpy library and tools from Pandas. Some additional tools that you may need to install are included at in the first few lines of the python script:

import re, sys, getopt, itertools, os, math
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from operator import truediv

# Using fastClines

## initial setup and study design

Prior to running fastClines the user needs to decide how the individuals (and their genomic data) will be organised along a 1-dimensional transect through a hybrid zone or steep transition zone. These usually involves the following steps:

1. select individuals that belong to a deme in a 2-dimensional landscape
2. set a linear (or polynomial) transect through the 2-dimensional landscape
3. collapse deme positions (mid-points) onto 1-D transect
4. for each 'i-th' deme, calculate the geographic span (di) of each deme along the transect.

This last step provides the information required for the 'spatial input data file' (see next section, files required)

## Files required

To run fastClines the following files are required:

### 1. fastClines script

### 2. scaffold data file (.txt format)

a data table listing the names of genomic input files to be expected for batch analyses of the scaffolds, chromosomes or linkage group to analyse. Its recommended that the genomic input is split into separate chromosomes to reduce output file sizes. 

Here is an example of the scaffold input data file:

| LG | Scaffold  | Scaffold Name | cM |
|----|-----------|--------------|----|
| 5  | Chr5_xaa | Chr5_xaa     | 1  |
| 5  | Chr5_xab | Chr5_xab     | 2  |
| 5  | Chr5_xac | Chr5_xac     | 3  |
| 5  | Chr5_xad | Chr5_xad     | 4  |
| 5  | Chr5_xae | Chr5_xae     | 5  |

In the example above, Chromosome number 5 (i.e. linkage group = LG) has been split into linear parts in order to split the large sync file into more manageable sections. 

To split files into smaller chunks this can be done easily with BASH 

e.g. > split -l 4000000 Chr6.sync

### 3. genomic input data file (.sync format) 

### 4. spatial input data file (.txt format) 

a file containing information about the size of the demes sampled across the transect. 

Here is an example of the spatial input datafile as used for the Antirrhinum data set

| Pool | N  | Dist | ID  | Lat       | Long      | Description        |
|------|----|------|-----|-----------|-----------|--------------------|
| 15   | 52 | 6000 | YP4 | 42.359921 | 1.926958  | A_m_striatum       |
| 16   | 50 | 5000 | YP2 | 42.325840 | 2.054797  | A_m_striatum       |
| 17   | 50 | 1500 | YP1 | 42.326943 | 2.052929  | A_m_striatum       |
| 18   | 50 | 1400 | MP2 | 42.323325 | 2.082989  | A_m_pseudomajus    |
| 19   | 50 | 3500 | MP4 | 42.322234 | 2.091375  | A_m_pseudomajus    |
| 20   | 50 | 6000 | MP11| 42.331038 | 2.170284  | A_m_pseudomajus    |

Note that the only columns of data used for FastClines are 'Pool' and 'Dist'. The latter is the deme span value (di)

The values must be non-zero and represent the 'span of each deme'. There can be a number of ways to do this depending on the spatial arrangement of the demes. If sets of individuals were sampled in discrete groups at regularly spaced intervals along the transect, the simplest approach is make all the deme spans equal values. However, often in hybrid zones, the spatial sampling of individuals and demes will be uneven. Therefore, one approach is the 'deme span' is equal to half the Euclidean distance between the average value of the samples in each deme. This assumes a linear 1D transect has been place through the centre of the hybrid zone and the perpendicular line from each deme to the 1D linear transect is then used to collapse 2D to 1D (see above).

## Running the fastClines script

Assuming the files required are prepared and in the same folder as the program script, an example of running fastClines is as follows: 

python3 fastClines_v1.4.py scaffold_Chr6_v3_5_ROSEL_testc.txt popDetails_Chr6_RosEL.txt clines_testc 10 300 2 6 0.8 1 0 1

in the command above, to run, after the script name you need the following arguments in this order: 

1. a txt file with a table of scaffolds to analyse 
2. a txt file with population names and spatial data
3. output file name
4. the minimum depth to include a site 
5. the max depth to include a site
6. minimum copies for an allele call (i.e. > 1, no singletons)
7. minimum number of populations (or pools) with minimum depth to include a site 
8. delta p (allele frequency difference) threshold to estimate cline parameters
9. how many of outer pools to use for delta p calculation (integer)
10. keep all sites (polymorphic or clinal only) (1 = polymorphic, 0 = clinal only)
11. poolSeq data (1 = Y,0 = N)

## Description of output for site output file

In the output file, the first four columns are fixed. Columns five onwards represent population specific values and pair-wise comparisons which use the population labels in spatial data file in the column headers.

- scaffold: genome scaffold/chromosome 
- position: position (site) on scaffold/chromosome in relation to reference genome
- ref: reference genome allele	
- bases_t: the nucleotides recorded at this position (site)
- p_i: allele frequency in population i through to N populations (i.e. number of columns for p depends on population number in the input file)
- dpthAdj_ith: raw depth (if individual genomes) or depth adjusted for poolSeq (see below) in population i through to N populations
- pi_i: nucleotide diversity in the ith to jth population
- centre: cline centre 
- width: cline width
- AFD_i_j: Allele Frequency Difference between ith and jth population
- piBar_i_j: average nucleotide diversity averaged across the ith and jth population pair
- piT_i_j: total nucleotide diversity combining ith and jth population pair
- dXYraw_i_j: absolute sequence divergence between ith and jth population calculated from allele frequencies
- dXYfromFst_i_j: as above, calculated from pi_T, Fst and pi_w (see Tavares et al., 2018) 	
-	FstfromPi_i_j: relative differentiation (Fst) between ith and jth population pair calculated from piBar, piT
-	FstfromDxy_i_j: relative differentiation (Fst) calculated from Dxy and piBar (see Tavares et al., 2018)

## additional information
- minimum depth: if running on individual based data, make minimum depth the size of the smallest number of haploid genomes (i.e. 4x2 = 8),
- minimum copies: if running on individual based data, make minimum copies = 1 (i.e. allow for singletons),
- input: scaffold input list must have line breaks saved as Unix (LF). Use text wrangler or similar text editor,
- in the sync files, populations must appear in the required geographic order along a transect/cline. 
- not all populations in the sync file have to be used, you can specificy only a subset of populations if required
- Geographic pop details file must list the numbers in consecutive order from the full set present in sync
 (e.g. in our example sync file, 20 pops are present but only pops 15,16,17,18,19,20 are utilised, as defined in the population file)
- Tavares, H., Whibley, A., Field, D. L., Bradley, D., Couchman, M., Copsey, L., Elleouet, J., Burrus, M., Andalo, C., Li, M., Li, Q., Xue, Y., Rebocho, A. B., Barton, N. H., & Coen, E. (2018). Selection and gene flow shape genomic islands that control floral guides. Proceedings of the National Academy of Sciences, 115(43), 11006–11011. https://doi.org/10.1073/pnas.1801832115

# Citing fastClines

If you use FastClines in any published work or implement the cline estimation method described here please cite:



# Authors and license information

David Field (david.field@mq.edu.au)

fastClines is available under the MIT license. See license.txt for more information
