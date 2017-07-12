PROGRAM: FAST

DESCRIPTION: Gene based whole-genome association analysis toolset

AUTHOR: Jianan Zhan

CONTACT: jianan.zhan@jhu.edu

YEAR: 2017

LICENSE: Released under GNU General Public License, v2 (see License.txt)


PREREQUISITES: GNU Scientific Library (GSL) is required to compile and use FAST. 
We have provided a copy of the GSL software in the GSL directory 
FAST.2.2.mc/GSL/gsl-1.15.tar.gz. GSL can also be downloaded from 
http://www.gnu.org/software/gsl/. Please unzip the .tar.gz file, it should 
create directory gsl-1.15. Please read the file INSTALL to install GSL, or follow 
the steps outlined in https://bitbucket.org/baderlab/fast/wiki/installGSL.

PRE-COMPILED BINARIES: We have provided pre-compiled binaries for Mac OSX and 
Linux x86_64 in the bin directory (FAST.1.6.mc/bin/) of the software.

For Linux
---------
The binary located at FAST.2.2.mc/bin/linux_x86_64/FAST is statically compiled 
meaning you should be able to run it just by ./FAST <options> without installing GSL. 
If you cannot run it, you need to install GSL and compile FAST. 
Please follow the instructions below for compilation.
Also you need to do ./install.sh to run scripts for combining output from several
gene based tests.

Mac OSX
-------
Mac OSX does not support static compilation. Please follow the instructions below 
for compilation. Also you need to do ./install.sh to run scripts for combining output 
from several gene based tests.


COMPILATION: You will need a standard C/C++ compiler such as GNU gcc
(version 3). This is likely available on all Linux/Unix platforms. For
MS-DOS, DJGPP or MinGW are appropriate choices. To compile for your platform
type "make clean" and then "make". 
- First install GSL by following the steps  mentioned in the INSTALL file 
  within GSL directory, or follow the steps in 
  https://bitbucket.org/baderlab/fast/wiki/installGSL.
- Once GSL is installed, download FAST and decompress the archive. Go to 
  the software home directory FAST.2.2.mc and type make clean and then type make. 
  This should compile the code to produce a binary for the appropriate platform. 
- Next run : ./install.sh. 
- The final executable is named FAST. Type ./FAST --help to see all the options.


USAGE: Type "FAST" or "./FAST" from the command line followed by the
options of choice (see documentation Documentation/Readme.pdf).

**** Important **** 
The included binary "FAST" is for linux x86_64. For running in Mac OSX 
,please replace this binary with the one from bin/macOS/FAST 
Otherwise, you will see the error : cannot execute binary file 
*******************

EXAMPLE DATA: Examples for both genotype data and summary data are provided 
in the Example directory. Example test scripts (.sh files) to run each are 
also provided.  Check these shell scripts to see how to run FAST for linear, 
logistic and summary modes.
 e.g ./run.Linear.geno.sh

SOURCE CODE: See Code/

ERROR: If you get any program crash / segmentation fault, please check the following:
# Check if the input/output options are properly specified with the --option flags (see Documentation)
# Check if the input / output files and paths are properly named. 
# Check if the input files have header lines (see Documentation for details) and follow the
  formatting described in Documentation/Readme.pdf.
# Check if the input files are TAB delimited and NOT space or other character delimited.
# Check if the input files do not have extra spaces / hidden characters / blank lines.

Others : Make sure that missing phenotype/covariate values are NA and NOT negative values.
Make sure that missing genotypes are a -ve number and the value is specified usin --missing-val 
option (default = -1).
