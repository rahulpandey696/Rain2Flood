# 🌧️ Rain2Flood: From Rainfall to Flood Inundation Mapping in QGIS
Changelog for Rain2Flood Plugin v2.1
🚀 Major New Features
Flash Flood Analysis (Hourly): Added specialized analysis for short-duration flood events using hourly rainfall data
Multiple Rainfall Data Sources:

Open-Meteo API integration for global hourly/daily data
Excel file support for custom rainfall data
Enhanced IMD and CHIRPS data processing
Enhanced DEM Processing: Improved catchment extraction and slope analysis
HEC-HMS Export: Generate input files for HEC-HMS hydrological modeling
Dependency Management: Automated package installation system

✨ Improvements
User Interface Enhancements:
Point selection directly from QGIS map
Calendar widgets for date selection
Dropdown suggestions for parameters (CN, Manning's n, runoff coefficients)

Analysis Methods:

SCS Unit Hydrograph method
Rational Method for short-duration storms
Time-Area Method

Flood Mapping:

Improved Manning's equation-based depth calculation
Better flood extent algorithms
Enhanced visualization
Memory Management: Optimized processing for large datasets

🐛 Bug Fixes
Fixed CHIRPS data extraction for global users
Improved frequency analysis for small datasets
Better error handling and user feedback

Fixed memory leaks in geopandas/rasterio operations

🔧 Technical Updates
Modular code structure with separate processing algorithms
Better exception handling and logging
Improved documentation and parameter descriptions
Enhanced plot visualizations


**Rain2Flood** is a powerful QGIS Processing Toolbox plugin that enables complete hydrological analysis — from rainfall data to flood inundation mapping — all within a single streamlined workflow.

Designed for researchers, students, water resource engineers, flood modellers, and disaster management professionals, it automates every step from rainfall frequency analysis to runoff calculation, hydrograph generation, and export to HEC-HMS formats.
Watch Video for the Workflow-

https://youtu.be/mJp9Be4vQcs?si=c-Tc5tqZuMlAX6oS

Follow on linkedin- www.linkedin.com/in/rahul-pandey-nitb
---
Nandi, Saswata, Pratiman Patel, and Sabyasachi Swain. "IMDLIB: An open-source library for retrieval, processing and spatiotemporal exploratory assessments of gridded meteorological observation datasets over India." Environmental Modelling & Software 171 (2024): 105869.
 
IMDLIB github repo - https://github.com/iamsaswata/imdlib
## 🚀 Key Features

- 🔄 **Automated Rainfall Processing**  
  Download and analyze daily rainfall data from **IMD** and **CHIRPS** directly within QGIS.

- 📊 **Frequency Analysis**  
  Fit rainfall distributions using **Gumbel**, **Log-Pearson III**, or **GEV** for custom return periods.

- 🌊 **Runoff Estimation**  
  Supports **Rational Method**, **SCS-CN**, and **Unit Hydrograph** approaches for peak discharge calculation.

- 🗺️ **Flood Inundation Mapping**  
  Generates **depth rasters** and **flood extent maps** using DEMs and flow path geometry.

- 💾 **HEC-HMS Integration**  
  Export hydrographs, rainfall events, and catchment parameters in HEC-HMS compatible formats.

- ⚙️ **Smart Parameter Parsing**  
  Intelligent handling of user inputs like Curve Number, Manning's n, and runoff coefficients with real-time hints.

---

## 🧪 Ideal For

- Hydrologists and water resource engineers  
- Disaster management professionals  
- GIS analysts and researchers  
- Students and academics studying flood modeling

---

## 🛠️ Getting Started

### 📦 Install Required Python Packages

Make sure you have QGIS with OSGeo4W Shell installed. Then run:


pip install imdlib xarray geopandas rasterio shapely scipy pandas numpy matplotlib
🧑‍💻 Contributing
We welcome all contributors — hydrologists, GIS specialists, and Python developers alike!

If you'd like to contribute, please:

Fork this repository

Create a new feature branch

Submit a pull request with a clear description
📜 License
This project is licensed under the GNU General Public License v3.0 or later (GPLv3+).
See the LICENSE file for details.

📢 Acknowledgements
Big thanks to:

IMD and CHIRPS for providing open-access rainfall datasets.

The IMDLIB team for enabling access to IMD gridded data in Python.

The global QGIS and open-source community.

🔗 Useful Links
📘 Official QGIS Plugin Repository (Submit/Track)






🌐 CHIRPS Rainfall Data

📘 IMDLIB Documentation

💻 QGIS Python Plugin Developer Guide
