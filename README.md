# GSL-RDASApp
RDASApp used in GSL only, including the GSIBEC recursive filter for regional use

RDASApp is cloned from git@github.com:Junjun-NOAA/RDASApp.git (June 26, 2025)   
Synchronized with https://github.com/NOAA-EMC/RDASApp.      
RDASApp checkout hash: 7a9a8fc29dafe11a94a8c2720f2dec92a570c2c1 (April 18, 2025)

Submodules with the following commit:   
gsibec: tag v1.3.1 (May 5, 2025)    
saber: hash 812d0ee1bc91f82716d121750674da0527cafec0 (April 4, 2025)    

And some modifications to the following files (updated with Masanori's copies)    
saber/gsi/grid/gsi_grid_mod.f90    
saber/gsi/covariance/gsi_covariance_mod.f90    
gsibec/gsi/control2state_ad.f90    
gsibec/gsi/control2state.f90    
gsibec/gsi/normal_rh_to_q.f90
gsibec/gsi/gsi_rfv3io_mod.f90 (multiple outloop)    
gsibec/gsi/gsibec/gsimod.F90 (new berror reading)
mpasjedi/LinearVariableChange/Control2Analysis/mpasjedi_linvarcha_c2a_mod.F90   
gsibec/gsi/mod_fv3_lola.f90











