# External Libaries

1) This software relies upon a few Python libraries. They are: numpy, healpy, scipy, sys, os, ConfigParser, astropy.io.

   Please ensure that they installed in your computer.

   If you want to ensure that, just type in a terminal:

      pip install -r requirements.txt --user

2) Please, ensure that you are running the latest version of Python2.7.

# Running the code

1) Just type in a terminal:

   python generate_foreground.py

2) You can change the parameters used to generate each foreground in parameters.ini.

3) The software functions are compiled in foregrounds_functions.py and misc_functions.py. 

# Output of the code

1) The code will output at least one 'fits' file with the total foreground cube (frequency vs Healpix pixels) in the directory output. You might choose extra outputs in the parameters file.

# Extra info

1) Please see the pdf file in the documentation directory. 

# Warning!

1) This is a work in progress. The software as it is now (May, 2018) is workable but still lacks development in terms of efficiency and generality (more models must be included).
