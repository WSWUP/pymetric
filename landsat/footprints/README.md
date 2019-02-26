The Landsat WRS2 Descending footprint shapefile can be downloaded directly from the [Landsat website](https://landsat.usgs.gov/pathrow-shapefiles) or using the [download script](tools/download/landsat_footprints.py) with the following command:

```
C:\pymetric>python tools\download\download_footprints.py
```

The file "wrs2_tile_utm_zones.json" was generated from the bulk metadata CSV files.  Path/rows that did not have any images in the metadata CSV files were removed.  Please refer to the [cloud-free-scene-counts repository](https://github.com/WSWUP/cloud-free-scene-counts) for additional details on acquiring the bulk metadata CSV files.
