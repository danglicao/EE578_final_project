Dangli Cao 9008679970 danglica@usc.edu
Renzhi Li  8282070004 renzhil@usc.edu

Project Title: Numerical Simulation of Half-Wave Dipole Antennas Using 3-D Finite-Difference Time-Domain Solver

If you want to run the code locally, please make sure you have a Nvidia GPU and install the environment in the environment.yml file.

Then you can gitclone the repo and run the code in dipole_with_traditional_gaussian.py

If you don't have a Nvidia GPU, we also provide a 1_ghz_antenna.ipynb. You can run it in Google Colab with their T4 GPU. You can directly run that file without install any environment.

If you want to compare with the open EMS, you can run the code in openMES_models, it set the same parameters as the dipole_with_traditional_gaussian.py. And remember to change the last line of the code to save the data in your local machine for comparison.
Then use our test_with_antenna.mlx to get the matlab data. You can run it in Matlab with the Antenna toolbox.
Then you can use the compare_Z.py to compare the real and imaginary part of the impedance from our model,matlab model and the openEMS model.

To easy the comparison, we also provide the impedance data from all three models in the data folder. You can directly use that data to compare with our model.