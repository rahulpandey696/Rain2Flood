# Rain2Flood
This QGIS Toolbox plugin processes rainfall data from IMD or CHIRPS sources to perform hydrological analysis. It calculates design rainfall for various return periods, estimates peak discharge, generates flood inundation maps, and produces hydrographs and HEC-HMS input files.
The Rain to Flood Analysis QGIS plugin enables end-to-end hydrological modeling by transforming rainfall data into flood inundation maps. It supports rainfall frequency analysis, runoff estimation, flood hydrograph generation, and automated outputs compatible with HEC-HMS and QGIS.

✨ Features
🛰️ Support for IMD and CHIRPS rainfall datasets

📈 Frequency analysis using Gumbel, Log-Pearson III, or GEV methods

💧 Runoff estimation via Rational Method, SCS-CN, and Unit Hydrograph

🗺️ Catchment area extraction from shapefile and DEM or manual input

⛰️ Automatic slope and flow path length calculation from raster/vector

🌊 Flood depth and inundation map generation

🧮 Time to Peak (Tp) calculation using SCS Lag Equation

📄 Exports results to Excel, HEC-HMS input files, hydrograph images, and flood maps

🧩 Fully integrated into QGIS Processing Toolbox

📦 Built-in dependency checker and OSGeo shell support
Requirements
QGIS 3.16 or later (tested on QGIS 3.40+)

Python packages: imdlib, xarray, geopandas, rasterio, shapely, scipy, pandas, numpy, matplotlib
