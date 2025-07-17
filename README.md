# Rain2Flood
Rain2Flood: From Rainfall to Flood Inundation Mapping in QGIS
This QGIS Toolbox plugin processes rainfall data from IMD or CHIRPS sources to perform hydrological analysis. It calculates design rainfall for various return periods, estimates peak discharge, generates flood inundation maps, and produces hydrographs and HEC-HMS input files.
The Rain to Flood Analysis QGIS plugin enables end-to-end hydrological modeling by transforming rainfall data into flood inundation maps. It supports rainfall frequency analysis, runoff estimation, flood hydrograph generation, and automated outputs compatible with HEC-HMS and QGIS.
Transform rainfall data into flood risk maps with a single workflow 🌧️→🌊

Rain2Flood is a powerful QGIS plugin that automates the entire hydrological analysis process - from downloading rainfall data to generating flood inundation maps. Designed for water resource engineers, flood modelers, and disaster management professionals, this tool integrates advanced hydrological methods within QGIS's user-friendly interface.

Key Features
Automated Rainfall Processing: Download and analyze IMD/CHIRPS rainfall data directly in QGIS

Hydrological Modeling: Implement Gumbel, Log-Pearson III, and GEV frequency distributions

Runoff Calculations: Support for Rational, SCS-CN, and Unit Hydrograph methods

Flood Mapping: Generate flood depth rasters and visual inundation maps

HEC-HMS Integration: Export inputs for advanced hydraulic modeling

Dynamic Parameter Handling: Smart parsing of hydrological parameters with context-aware suggestions
Getting Started
Install required Python packages:
pip install imdlib xarray geopandas rasterio shapely scipy pandas numpy matplotlib

Place the processing_provider.py file in your QGIS plugins directory

Activate the plugin through QGIS Processing Toolbox
Contributing
We welcome contributions! Please see our contribution guidelines for details. Whether you're a hydrologist, GIS specialist, or Python developer, your expertise can help improve flood modeling for everyone.

License
This project is licensed under GPLv3 - see the LICENSE file for details.

Python packages: imdlib, xarray, geopandas, rasterio, shapely, scipy, pandas, numpy, matplotlib
