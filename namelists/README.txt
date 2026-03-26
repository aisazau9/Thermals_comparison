This folder contains the namelist files used to run the WRF simulations for this project.

Contents

Each subfolder corresponds to one event (case) and contains the namelist files required to run WRF:

namelist.wps – configuration for WPS (domain setup, geogrid, ungrib, metgrid)
namelist.input – configuration for WRF model integration
dt_values_{CASE}.txt - as the timestep was (manually) reduced during model integration, this textfile contains the timestep used to generate each output (d01)

