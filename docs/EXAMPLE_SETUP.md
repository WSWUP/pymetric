Quicklinks: [Data Preparation](EXAMPLE_DATA.md) --- [Project Setup](EXAMPLE_SETUP.md) --- [Running METRIC](EXAMPLE_METRIC.md)

# pyMETRIC Project Setup Example

This example will step through setting up and running pyMETRIC for a single Landsat path/row.  The target study area for this example is the Harney basin in central Oregon, located in Landsat path 43 row 30.  Before going through this example, make sure that the Landsat images and ancillary data have been acquired and prepped following the steps in the [Setup Example](EXAMPLE_SETUP.md).

## Project Folder

All of the example script calls listed below assume that the pyMETRIC repository was installed on a Windows computer to "D:\pyMETRIC", that a "Harney" project folder was created in this folder, and that the scripts are being called from within the project folder (see [Setup Example](EXAMPLE_SETUP.md)).  If you haven't already, change directory into the Harney project folder.

```
D:\pyMETRIC>cd harney
```

## INI

Need to explain the difference between the INIs.  The project INI is the main INI the user will change when running pyMETRIC.  The values in the project INI are passed to the script specific INIs using the prep INI tool, but they can also be manually changed/overwritten by the user.

## Project INI

Copy the template landsat_project.ini from the code\ini_templates folder to the Harney folder.

Rename the template INI to "landsat_2015.ini" using the command line or file explorer.  Typically, a separate INI file will be needed for each year that is processed.

After renaming the INI, open the INI file in your favorite text editor.  The default values for the template INI file have been set for the Harney Basin for 2015.  To use the template INI in a different study area or folder structure, it will be necessary to change all of the folder paths from "D:\pyMETRIC\harney" to the new project folder.

### Snap Point

The study_area_snap and zones_snap parameters should both be set to 15, 15 in the project INI.  This ensures that alignment of the final ET maps and zonal statistics calculations will align with a Landsat image of the study area.  The snap points will typically be set to 0, 0 if the final image is in a non-WGS84 Zone XX (EPSG:326XX) coordinate system and doesn't not need to align with a Landsat image.

### Fields

A field polygon dataset is not currently being provided or available for this example  The cropland data layer (CDL) could be used to generated quasi field boundaries if desired.  For now, the user should ensure that the following flags in the project INI are all false: "cdl_flag", "landfire_flag", "field_flag".

## pyMETRIC Setup

### Prep path/row ancillary data

This script will unpack the Landsat scenes and create the ancillary datasets for each Landsat path/row.

```
D:\pyMETRIC\harney>python ..\code\local\landsat_prep_path_row.py -i landsat_2015.ini
```

### Prep INI files

This script will generate the INI files for all of the subsequent processes.  This script can also be used to update the INI files if the users makes a change to the main project INI.

```
D:\pyMETRIC\harney>python ..\code\local\landsat_prep_ini.py -i landsat_2015.ini
```

### Prep Landsat scenes

```
D:\pyMETRIC\harney>python ..\code\local\landsat_prep_scene.py -i landsat_2015.ini
```
