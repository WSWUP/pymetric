# pyMETRIC

pyMETRIC is a set of Python based tools developed for estimating and mapping evapotranspiration (ET) for large areas, utilizing the Landsat image archive.  This framework currently computes ET estimates using the [METRIC](http://www.uidaho.edu/cals/kimberly-research-and-extension-center/research/water-resources) surface energy balance model, developed at the University of Idaho.
 
In order to produce ET estimates, pyMETRIC produces ancillary rasters from Landsat data products.  These products are stored within the pyMETRIC data structure, and may be useful for tasks tangentially related to ET mapping. The raster datasets produced during typical processing include the following:
- Albedo
- LAI (Leaf Area Index)
- NDVI (Normalized Difference Vegetation Index)
- NDWI (Normalized Difference Water Index)
- Top of Atmosphere Reflectance

In addition to creating ET maps from Landsat images, pyMETRIC includes functionality to interpolate annual/seasonal/monthly ET maps, from individually processed ET maps.

## Install

Details on installing pyMETRIC, Python, and necessary modules can be found in the[ installation instructions](docs/INSTALL.md).

## Example

A detailed walk-through on the setup and operation of pyMETRIC has been assembled in the following series of documentation.  These examples are setup to process a portion of the Harney Basin, located in eastern Oregon.  The documentation is contained in the following links:
1. [Data Preparation](docs/EXAMPLE_DATA.md)
2. [Project Setup](docs/EXAMPLE_SETUP.md)
3. [Running METRIC](docs/EXAMPLE_METRIC.md)

## References

* [Satellite-Based Energy Balance for Mapping Evapotranspiration with Internalized Calibration (METRIC)—Model](https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9437(2007)133:4(380))
* [Satellite-Based Energy Balance for Mapping Evapotranspiration with Internalized Calibration (METRIC)—Applications](https://ascelibrary.org/doi/abs/10.1061/(ASCE)0733-9437(2007)133:4(395))
* [Assessing calibration uncertainty and automation for estimating evapotranspiration from agricultural areas using METRIC](https://www.dri.edu/images/stories/divisions/dhs/dhsfaculty/Justin-Huntington/Morton_et_al._2013.pdf)

## Limitations

METRIC requires an assemblage of several datasets in order to produce accurate estimates of evapotranspiration.  The pyMETRIC framework serve to download and process the required data.  Please note that this code is written for the data as it is currently provided, however the data and it’s formatting is controlled by the data providers and by third-party hosts.  The maintainers of pyMETRIC will attempt to keep the package functional, however changes in the data and data availability may impact the functionality of pyMETRIC.

## Directory Structure

When initially downloading or cloning pyMETRIC, this directory does not contain data necessary for estimating ET.  As  python scripts are ran as prescribed in ["Data Preparation"](docs/EXAMPLE_DATA.md) and ["Project Setup"](docs/EXAMPLE_SETUP.md), the top level directory will be populated with additional directories containing support data. These folders will be assigned names according to the directory contents (eg. "cdl", "dem", "gridmet", etc...).  Ideally these data directories will be populated with project-agnostic data (example. "dem" may contain a digital elevation model (DEM) for the entire continental United States).  The support data will be processed by pyMETRIC, which will isolate and subset the relevant data for processing. 

To serve as an example the "harney" directory is included in the top-level directory.  The "harney" directory is an example of a "project directory", which should contain information specific to your chosen study area or area of interest.   As pyMETRIC is ran according to ["Running METRIC"](docs/EXAMPLE_METRIC.md), the processed data will be stored within the project directory.