# 🌧️ Rain2Flood: From Rainfall to Flood Inundation Mapping in QGIS

**Rain2Flood** is a powerful QGIS Processing Toolbox plugin that enables complete hydrological analysis — from rainfall data to flood inundation mapping — all within a single streamlined workflow.

Designed for researchers, students, water resource engineers, flood modellers, and disaster management professionals, it automates every step from rainfall frequency analysis to runoff calculation, hydrograph generation, and export to HEC-HMS formats.
Watch Video for the Workflow-
https://youtu.be/H5JiZJiFRd8?si=uoVIoUiu2pwevTx3
Follow on linkedin- www.linkedin.com/in/rahul-pandey-nitb
---

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

```bash
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
