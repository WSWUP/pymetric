# Harney Basin Example

To serve as an example the "harney" directory is included with pyMETRIC.  The "harney" directory is an example of a "project directory", which should contain information specific to your chosen study area or area of interest.   As pyMETRIC is ran according to ["Running METRIC"](docs/EXAMPLE_METRIC.md), processed data will be stored within this project directory.  

When downloading support data, processing intermediate products, and producing ET data, the pyMETRIC code will allow you to define the location of support data and products.  It's highly reccomended that you use the default paths for the pyMETRIC code.  Deviation from this directory structure may result in conflicts between the various steps of the pyMETRIC workflow. An exapmple of the directory structure for the Harney example can be found in ["Harney_Example_Directory_Structure.txt"](harney/Harney_Example_Directory_Structure.txt)


## landsat_2015.ini
This file serves as the project input file (INI).  This file contains necessary inputs needed to running the pyMETRIC process.  Most python scripts within the METRIC workflow will reference this specific file.  This file is already populated with the necessary inputs to run the pyMETRIC example.

## landsat
When pyMETRIC is downloaded or cloned, the only contents of this directory will be "skip_list.txt."  This file stores Landsat scenes that deemed by the user to not be useful in the pyMETRIC processes.  A scene is most often skipped when the specific Landsat scene contains clouds that obscure the study area.

## study_area
This file contains a shapefile (Esri vector data storage format) the delineates the study area where ET estimates are to be calculated.  This shapefile will used to subset support data and limit processing only to the areas where data is desired.  In this example, the shapefile was derived from the USGS National Hydrography Dataset (WBDHU8) for a hydrographic basin in the area of the Harney, Oregon.

![Alt text](../docs/images/harney_shapefile.png?raw=true "Harney Basin, Oregon")


