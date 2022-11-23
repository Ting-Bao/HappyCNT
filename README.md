# HappyCNT

Do happy Wannierization and postprocessing of Carbon NanoTube (CNT), Hybrid Organic-Inorganic Perovskit (HOIP), etc. The code is powered by Ting BAO and Benshu FAN.

## wannier scan part (CNT_wannierhelper/)
This part is to scan the inner window of the wannierization.
+ gen_projector.py: 
Generate the projector part of wannier.win file. This is only applicable for CNT, moving the center line of CNT along the (0,0,1) of the unitcell and generate intial orbitals as follows:
    - p_z orbital on each atom, pointing out from the center 
    - mid-bond s orbit
+ scan_window.py:
    Read in the config.ini file and generate wannierization input file in batch according to **[dis_froz_min]** and **[dis_froz_max]**, see example config.ini for details. Also, the feimi energy should be given (can be grepped from VASP Calculation). In the templated path, files following should be provided:
    - a config file gives out the inner step range, see scan_config.ini
    - a template of wannier90.win, with 
        +  dis_froz_min = CONTENT1
        +  dis_froz_max = CONTENT2
        + a template run.sh of wannier90.x
        + data file from VASP projection: wannier90.{amn,eig,mmn}
        + wannier plot code from fanbs
        + BAND.dat, KLABELS


## The postprocess part (postprocess/)
This part aims to get the band structure from the wannierized model and add the effect of magnetic flux.


## TODO list
+ functionalize the postprocess code (Ting BAO finished!)
+ add the Berry Phase related part:
    + Calculate the Berry curvatrue using Wilson Loop/Normal Berry curvature formula
    + Calculate the mag-related parameter, $\chi(\phi)$ 
    + Calculate the Chern number/Berry phase
