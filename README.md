Code for Uzsoy & Villar (2026): "Photometry is all you need: supernova classification as a mixing problem". Note that this repository includes code written with GitHub CoPilot and Claude Code. The name "beams-ext" comes from our aim to extend on the BEAMS framework presented in Kunz et al. (2007).

A quick tour:

- `fit_real_sne_new.ipynb` has all of the code to fit SN light curves as both SNe Ia and Ibc. This is wrapped into `fit_real_sne_new.py`, which is a script to quickly fit all light curves from the command line.
- `separate_parameters.ipynb` has all of the code to do the joint GMM fit using all 8 parameters (that we don't end up using) and individual 2D GMM fits.
- `compare_w_redshifts.ipynb` runs 2D GMM fits for every permutation of redshift estimate and with + without 10% known classifications. 

This is the only code used to actually run the analysis and make the plots in the paper, everything else was intermediate steps it took to get there!
