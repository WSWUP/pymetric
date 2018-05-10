# NLDAS

Scripts for downloading and preparing NLDAS hourly weather data
        Hourly vapour pressure data is used to generate the Tasumi at-surface reflectance data
        Hourly wind speed and ASCE standardized reference ET (ETr) are used in METRIC to estimate ET

nldas_ancillary.py - Download and process the NLDAS mask, elevation, 
    latitude, and longitude rasters
nldas_download.py - Download the ".grb" files from the NLDAS website
    The default date range is 2017-01-01 to 2017-12-31
nldas_hourly_ea.py - Calculate hourly vapour pressure from the NLDAS inputs
nldas_hourly_wind.py - Calculate hourly wind speed from the NLDAS inputs  
nldas_hourly_refet.py - Calculate hourly ETo and ETr from the NLDAS inputs
    ETo and ETr are saved as hourly IMG rasters in separate folders

##Run nldas_hourly_variable.py to extract NLDAS variable(s)
## Not currently supported
