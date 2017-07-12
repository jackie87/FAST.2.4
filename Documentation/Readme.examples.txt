This Readme file describes how to run the examples included with FAST.
All the example data and output are located in ./Example directory.

  -- To run FAST with Linear regression, follow these steps:
  a) cd Example
  b) ./run.Linear.geno.sh
  
  -- If you get the error "cannot execute binary file", replace the binary file 'FAST' in 
  FAST_home_directory (i.e the directory containing the folders Code/ , Documentation/ ,
  Example/ etc.) with the one suitable for your platform from ./bin/. 

  e.g. if you are in Mac OS, do : 
       cd <your FAST_home_directory>
       cp bin/macOS/FAST .
       repeat step a) and b).

  If you still get the error "cannot execute binary file", you may need to compile FAST. 
  See COMPILATION in Readme.txt

  -- If you have successfully completed (a) and (b), you can prepare a combined report
  from all methods as (you must have done ./install.sh from your FAST_home_directory prior
  to this): 

  c) cd <your FAST_home_directory>
  d) ./FAST.utils.sh -r ./Example/OUT/output Linear 0.0001 0.0001
  
  This will generate a report file 'report.Linear.txt'.
  NOTE : The chromosome number 'chrxxx' is EXCLUDED in the 2nd argument in step (d)
  as 'chrxxx' is automatically added by FAST to your chosen output file name, and
  the script to generate the combined report will iterate over all such chromosomes.

  You can run for type Logistic or Summary similarly, just replace the file in step (b)
  with the appropriate file, see inside Example/.

  -- Alternatively, if you have successfully completed (a) and (b), you can prepare qq plots
  from all methods as (you must have done ./install.sh from your FAST_home_directory prior
  to this) : 

  c) cd <your FAST_home_directory>
  d) ./FAST.utils.sh -p ./Example/OUT/output Linear
  
  This will generate jpeg files for each method.
  NOTE : The chromosome number 'chrxxx' is EXCLUDED in the 2nd argument in step (d)
  as 'chrxxx' is automatically added by FAST to your chosen output file name, and
  the script to generate qq plots will iterate over all such chromosomes.

  You can run for type Logistic or Summary similarly, just replace the file in step (b)
  with the appropriate file, see inside Example/.

  -- To see how to run the utility functions, do the following : 
  a) cd <your FAST_home_directory>
  b) ./FAST.utils.sh
