import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import math
import gc
import traceback
import importlib
import subprocess
from pathlib import Path

# QGIS Processing imports
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingOutputFile,
    QgsProcessingOutputFolder,
    QgsProcessingProvider,
    QgsProject,
    QgsProcessingException,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPointXY,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterVectorLayer,
    QgsRasterLayer,
    QgsFeatureRequest,
    QgsColorRampShader,
    QgsSingleBandPseudoColorRenderer,
    QgsRasterShader,
    QgsRasterMinMaxOrigin,
    QgsRasterBandStats
)
from qgis.PyQt.QtGui import QIcon, QColor
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QMessageBox, QPushButton

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Define required packages
REQUIRED_PACKAGES = [
    'imdlib', 'xarray', 'geopandas', 'rasterio', 
    'shapely', 'scipy', 'pandas', 'numpy', 'matplotlib'
]

class DependencyManager:
    @staticmethod
    def check_dependencies():
        """Check if all required packages are installed"""
        missing = []
        for package in REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)
        return missing
    
    @staticmethod
    def show_install_dialog(parent=None):
        """Show installation instructions dialog"""
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Required Dependencies")
        msg.setText(
            "The Rain2Flood plugin requires additional Python packages.\n\n"
            "Please install them using the OSGeo4W shell:\n\n"
            "1. Open OSGeo4W Shell (search in Start menu)\n"
            "2. Run this command:\n"
            "   pip install imdlib xarray geopandas rasterio shapely scipy pandas numpy matplotlib\n\n"
            "After installation, restart QGIS and try again."
        )
        
        # Add buttons
        open_button = QPushButton("Open OSGeo4W Shell")
        copy_button = QPushButton("Copy Command")
        msg.addButton(open_button, QMessageBox.ActionRole)
        msg.addButton(copy_button, QMessageBox.ActionRole)
        msg.addButton(QMessageBox.Ok)
        
        # Execute dialog
        msg.exec_()
        
        # Handle button clicks
        if msg.clickedButton() == open_button:
            DependencyManager.open_osgeo_shell()
        elif msg.clickedButton() == copy_button:
            DependencyManager.copy_command_to_clipboard()
    
    @staticmethod
    def open_osgeo_shell():
        """Open OSGeo4W Shell on Windows"""
        try:
            # Common installation paths
            paths = [
                r"C:\OSGeo4W\OSGeo4W.bat",
                r"C:\Program Files\QGIS 3.34.0\OSGeo4W.bat",
                r"C:\Program Files\QGIS 3.32.0\OSGeo4W.bat",
                r"C:\Program Files\QGIS 3.30.0\OSGeo4W.bat",
            ]
            
            # Check which path exists
            valid_path = None
            for path in paths:
                if os.path.exists(path):
                    valid_path = path
                    break
            
            if valid_path:
                subprocess.Popen(['start', valid_path], shell=True)
            else:
                QMessageBox.warning(
                    None, 
                    "OSGeo4W Shell Not Found",
                    "Could not find OSGeo4W Shell. Please open it manually from the Start menu."
                )
        except Exception as e:
            QMessageBox.critical(
                None, 
                "Error Opening Shell",
                f"Failed to open OSGeo4W Shell: {str(e)}"
            )
    
    @staticmethod
    def copy_command_to_clipboard():
        """Copy install command to clipboard"""
        try:
            import pyperclip
            command = "pip install imdlib xarray geopandas rasterio shapely scipy pandas numpy matplotlib"
            pyperclip.copy(command)
            QMessageBox.information(
                None, 
                "Command Copied",
                "Install command copied to clipboard!"
            )
        except ImportError:
            # Fallback if pyperclip not available
            QMessageBox.information(
                None, 
                "Command",
                "Run this command in OSGeo4W Shell:\n\n" + command
            )

class FeedbackStream:
    def __init__(self, feedback):
        self.feedback = feedback
    
    def write(self, text):
        if text.strip():
            self.feedback.pushInfo(text)
    
    def flush(self):
        pass

class Rain2FloodAlgorithm(QgsProcessingAlgorithm):
    # Algorithm parameters
    COORDINATES = 'COORDINATES'
    YEAR_RANGE = 'YEAR_RANGE'
    RETURN_PERIODS = 'RETURN_PERIODS'
    RAINFALL_SOURCE = 'RAINFALL_SOURCE'
    IMD_FOLDER = 'IMD_FOLDER'
    CHIRPS_FOLDER = 'CHIRPS_FOLDER'
    FREQ_METHOD = 'FREQ_METHOD'
    RUNOFF_METHOD = 'RUNOFF_METHOD'
    CATCHMENT_INPUT = 'CATCHMENT_INPUT'
    SHAPEFILE = 'SHAPEFILE'
    DEM = 'DEM'
    SLOPE_RASTER = 'SLOPE_RASTER'
    FLOW_PATH = 'FLOW_PATH'
    AREA = 'AREA'
    SLOPE = 'SLOPE'
    CN = 'CN'
    RUNOFF_COEFF = 'RUNOFF_COEFF'
    MANNING_N = 'MANNING_N'
    OUTPUT_DIR = 'OUTPUT_DIR'
    GENERATE_MAPS = 'GENERATE_MAPS'
    LOAD_RASTERS = 'LOAD_RASTERS'
    EXPORT_HEC_HMS = 'EXPORT_HEC_HMS'

    # Outputs
    OUTPUT_EXCEL = 'OUTPUT_EXCEL'
    OUTPUT_FLOOD_MAPS = 'OUTPUT_FLOOD_MAPS'
    OUTPUT_FLOOD_RASTERS = 'OUTPUT_FLOOD_RASTERS'
    OUTPUT_HYDROGRAPHS = 'OUTPUT_HYDROGRAPHS'
    OUTPUT_HEC_HMS = 'OUTPUT_HEC_HMS'

    # CN value suggestions
    CN_VALUES = [
        '40 (Woods)', '45 (Farmland)', '50 (Pasture)', 
        '55 (Brush)', '60 (Residential)', '65 (Agricultural)',
        '70 (Agricultural)', '75 (Urban)', '80 (Commercial)',
        '85 (Industrial)', '90 (Paved areas)', '95 (Waterproof)'
    ]
    
    # Runoff coefficient suggestions
    RUNOFF_COEFF_VALUES = [
        '0.10 (Sandy soil)', '0.15 (Forest)', '0.20 (Grassland)',
        '0.30 (Cultivated land)', '0.40 (Residential)', '0.50 (Suburban)',
        '0.60 (Urban)', '0.70 (Commercial)', '0.80 (Industrial)',
        '0.90 (Paved areas)'
    ]
    
    # Manning's n value suggestions
    MANNING_N_VALUES = [
        '0.01 (Smooth concrete)', '0.02 (Finished concrete)', 
        '0.03 (Earth channel)', '0.04 (Gravel channel)', 
        '0.05 (Natural stream)', '0.06 (Weedy stream)', 
        '0.07 (Dense vegetation)', '0.08 (Floodplain)', 
        '0.10 (Heavy brush)', '0.12 (Forest)'
    ]

    def initAlgorithm(self, config=None):
        # Check dependencies
        missing = DependencyManager.check_dependencies()
        if missing:
            DependencyManager.show_install_dialog()
            raise QgsProcessingException(
                f"Missing required packages: {', '.join(missing)}. "
                "Please install them using OSGeo4W Shell and restart QGIS."
            )
        
        # Combined coordinate input
        self.addParameter(QgsProcessingParameterString(
            self.COORDINATES, 'Coordinates (Latitude, Longitude)',
            defaultValue='20.0, 77.0'))
        
        # Combined year range input
        self.addParameter(QgsProcessingParameterString(
            self.YEAR_RANGE, 'Year Range (start-end)',
            defaultValue='1980-2020'))
        
        self.addParameter(QgsProcessingParameterString(
            self.RETURN_PERIODS, 'Return Periods (comma separated)',
            defaultValue='10,25,50,100'))
        
        self.addParameter(QgsProcessingParameterEnum(
            self.RAINFALL_SOURCE, 'Rainfall Source',
            options=['IMD', 'CHIRPS'], defaultValue=0))
        
        self.addParameter(QgsProcessingParameterFile(
            self.IMD_FOLDER, 'IMD Data Folder (for IMD source)',
            behavior=QgsProcessingParameterFile.Folder, optional=True))
        
        self.addParameter(QgsProcessingParameterFile(
            self.CHIRPS_FOLDER, 'CHIRPS Data Folder (for CHIRPS source)',
            behavior=QgsProcessingParameterFile.Folder, optional=True))
        
        self.addParameter(QgsProcessingParameterEnum(
            self.FREQ_METHOD, 'Frequency Analysis Method',
            options=['Gumbel', 'Log-Pearson III', 'GEV'], defaultValue=0))
        
        self.addParameter(QgsProcessingParameterEnum(
            self.RUNOFF_METHOD, 'Runoff Calculation Method',
            options=['Rational', 'SCS-CN', 'Unit Hydrograph'], defaultValue=1))
        
        self.addParameter(QgsProcessingParameterEnum(
            self.CATCHMENT_INPUT, 'Catchment Input Method',
            options=['Shapefile & DEM', 'Manual Parameters'], defaultValue=0))
        
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.SHAPEFILE, 'Catchment Boundary',
            types=[QgsProcessing.TypeVectorPolygon],
            optional=True))
        
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.DEM, 'DEM Raster',
            optional=True))
            
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.SLOPE_RASTER, 'Slope Raster (percent)',
            optional=True))
        
        self.addParameter(QgsProcessingParameterVectorLayer(
            self.FLOW_PATH, 'Longest Flow Path (Line)',
            types=[QgsProcessing.TypeVectorLine],
            optional=True))
        
        self.addParameter(QgsProcessingParameterNumber(
            self.AREA, 'Catchment Area (km²)',
            QgsProcessingParameterNumber.Double, optional=True, minValue=0.1))
        
        self.addParameter(QgsProcessingParameterNumber(
            self.SLOPE, 'Mean Slope (decimal)',
            QgsProcessingParameterNumber.Double, optional=True, minValue=0.001))
        
        # CN with manual input and dropdown suggestions
        self.addParameter(QgsProcessingParameterString(
            self.CN, 'Curve Number',
            defaultValue='65 (Agricultural)'))
        
        # Runoff coefficient with manual input and dropdown suggestions
        self.addParameter(QgsProcessingParameterString(
            self.RUNOFF_COEFF, 'Runoff Coefficient (for Unit Hydrograph)',
            defaultValue='0.40 (Residential)',
            optional=True))
            
        # Manning's n with manual input and dropdown suggestions
        self.addParameter(QgsProcessingParameterString(
            self.MANNING_N, "Manning's Roughness Coefficient (for flood mapping)",
            defaultValue='0.05 (Natural stream)'))
        
        self.addParameter(QgsProcessingParameterBoolean(
            self.GENERATE_MAPS, 'Generate Flood Maps', defaultValue=True))
        
        self.addParameter(QgsProcessingParameterBoolean(
            self.LOAD_RASTERS, 'Load output flood rasters in QGIS', defaultValue=True))
        
        self.addParameter(QgsProcessingParameterBoolean(
            self.EXPORT_HEC_HMS, 'Export HEC-HMS Input Files', defaultValue=True))
        
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIR, 'Output Directory'))
        
        # Outputs
        self.addOutput(QgsProcessingOutputFile(
            self.OUTPUT_EXCEL, 'Results Excel File'))
        
        self.addOutput(QgsProcessingOutputFolder(
            self.OUTPUT_FLOOD_MAPS, 'Flood Maps Folder (PNG)'))
            
        self.addOutput(QgsProcessingOutputFolder(
            self.OUTPUT_FLOOD_RASTERS, 'Flood Rasters Folder (GeoTIFF)'))
            
        self.addOutput(QgsProcessingOutputFolder(
            self.OUTPUT_HYDROGRAPHS, 'Hydrographs Folder (PNG)'))
            
        self.addOutput(QgsProcessingOutputFolder(
            self.OUTPUT_HEC_HMS, 'HEC-HMS Input Files Folder'))

    def processAlgorithm(self, parameters, context, feedback):
        try:
            # Check dependencies again in case user ignored first warning
            missing = DependencyManager.check_dependencies()
            if missing:
                DependencyManager.show_install_dialog()
                raise QgsProcessingException(
                    f"Missing required packages: {', '.join(missing)}. "
                    "Please install them using OSGeo4W Shell and restart QGIS."
                )
            
            # Import required modules after dependency check
            import imdlib as imd
            import xarray as xr
            import geopandas as gpd
            import rasterio
            from rasterio.mask import mask
            from rasterio import features
            from shapely.geometry import mapping
            from rasterio.plot import plotting_extent
            from scipy.stats import genextreme, gumbel_r, pearson3
            from scipy import ndimage
            
            # Get coordinates from combined input
            coord_str = self.parameterAsString(parameters, self.COORDINATES, context)
            try:
                parts = coord_str.split(',')
                lat = float(parts[0].strip())
                lon = float(parts[1].strip())
            except:
                raise QgsProcessingException("Invalid coordinate format. Use: 'latitude, longitude'")
            
            feedback.pushInfo(f"Using coordinates: {lat:.6f}°N, {lon:.6f}°E")

            # Validate coordinates
            if abs(lat) < 0.001 and abs(lon) < 0.001:
                raise QgsProcessingException("Invalid coordinates (0,0) - please provide valid location")

            # Get year range from combined input
            year_range_str = self.parameterAsString(parameters, self.YEAR_RANGE, context)
            try:
                parts = year_range_str.split('-')
                start_year = int(parts[0].strip())
                end_year = int(parts[1].strip())
            except:
                raise QgsProcessingException("Invalid year range format. Use: 'start-end'")
            
            # Get other parameters
            return_periods = [int(rp) for rp in self.parameterAsString(parameters, self.RETURN_PERIODS, context).split(',')]
            rainfall_source = self.parameterAsEnum(parameters, self.RAINFALL_SOURCE, context) + 1
            imd_folder = self.parameterAsString(parameters, self.IMD_FOLDER, context)
            chirps_folder = self.parameterAsString(parameters, self.CHIRPS_FOLDER, context)
            freq_method = self.parameterAsEnum(parameters, self.FREQ_METHOD, context) + 1
            runoff_method = self.parameterAsEnum(parameters, self.RUNOFF_METHOD, context) + 1
            catchment_input = self.parameterAsEnum(parameters, self.CATCHMENT_INPUT, context)
            load_rasters = self.parameterAsBoolean(parameters, self.LOAD_RASTERS, context)
            export_hec_hms = self.parameterAsBoolean(parameters, self.EXPORT_HEC_HMS, context)
            
            # Process CN value
            cn_input = self.parameterAsString(parameters, self.CN, context).strip()
            cn_value = self.parse_value_from_input(cn_input, self.CN_VALUES, "CN", 65, feedback)
            feedback.pushInfo(f"Using Curve Number: {cn_value}")
            
            # Process runoff coefficient
            runoff_coeff_input = self.parameterAsString(parameters, self.RUNOFF_COEFF, context)
            if runoff_coeff_input:
                runoff_coeff_input = runoff_coeff_input.strip()
                runoff_coeff_value = self.parse_value_from_input(
                    runoff_coeff_input, self.RUNOFF_COEFF_VALUES, "Runoff Coefficient", 0.4, feedback
                )
                feedback.pushInfo(f"Using Runoff Coefficient: {runoff_coeff_value}")
            else:
                runoff_coeff_value = 0.4
                feedback.pushInfo("Using default Runoff Coefficient: 0.4")
            
            # Process Manning's n
            manning_n_input = self.parameterAsString(parameters, self.MANNING_N, context).strip()
            manning_n_value = self.parse_value_from_input(
                manning_n_input, self.MANNING_N_VALUES, "Manning's n", 0.05, feedback
            )
            feedback.pushInfo(f"Using Manning's n: {manning_n_value}")
            
            # Get layers from project
            shapefile_layer = self.parameterAsVectorLayer(parameters, self.SHAPEFILE, context)
            dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
            slope_raster_layer = self.parameterAsRasterLayer(parameters, self.SLOPE_RASTER, context)
            flow_path_layer = self.parameterAsVectorLayer(parameters, self.FLOW_PATH, context)
            
            shapefile_path = shapefile_layer.source() if shapefile_layer else None
            dem_path = dem_layer.source() if dem_layer else None
            slope_raster_path = slope_raster_layer.source() if slope_raster_layer else None
            flow_path = flow_path_layer.source() if flow_path_layer else None
            
            area = self.parameterAsDouble(parameters, self.AREA, context)
            slope = self.parameterAsDouble(parameters, self.SLOPE, context)
            generate_maps = self.parameterAsBoolean(parameters, self.GENERATE_MAPS, context)
            output_dir = self.parameterAsString(parameters, self.OUTPUT_DIR, context)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Redirect stdout
            original_stdout = sys.stdout
            sys.stdout = FeedbackStream(feedback)
            
            # Run analysis
            feedback.pushInfo("Starting hydrological analysis...")
            feedback.pushInfo(f"Output directory: {output_dir}")

            # ===== CATCHMENT PARAMETERS =====
            area_m2 = None
            dem_profile = None
            slope_percent = None
            longest_flow_path = None
            
            if catchment_input == 0:  # Shapefile & DEM
                if not shapefile_path or not dem_path:
                    raise QgsProcessingException("Both catchment boundary and DEM are required")
                
                if not flow_path:
                    raise QgsProcessingException("Longest flow path vector is required for automated method")
                    
                if not slope_raster_path:
                    raise QgsProcessingException("Slope raster is required for automated method")
                
                try:
                    catchment = gpd.read_file(shapefile_path)
                    if catchment.crs.is_geographic:
                        centroid = catchment.geometry.unary_union.centroid
                        utm_zone = int((centroid.x + 180) // 6 + 1)
                        hemisphere = 'north' if centroid.y >= 0 else 'south'
                        crs_utm = f"+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
                        catchment = catchment.to_crs(crs_utm)
                    area_m2 = catchment.geometry.area.sum()
                    area = area_m2 / 1e6
                    feedback.pushInfo(f"Computed catchment area: {area:.3f} km²")
                except Exception as e:
                    raise QgsProcessingException(f"Error computing area: {str(e)}")

                # Calculate slope from slope raster
                try:
                    with rasterio.open(slope_raster_path) as src:
                        catchment_slope = catchment.to_crs(src.crs)
                        geoms = [mapping(geom) for geom in catchment_slope.geometry]
                        
                        out_image, out_transform = mask(src, geoms, crop=True, nodata=np.nan)
                        slope_data = out_image[0]
                        
                        valid_slope = slope_data[(slope_data > 0) & (~np.isnan(slope_data))]
                        if len(valid_slope) == 0:
                            feedback.pushWarning("No valid slope values found in catchment. Using manual slope input.")
                            if not slope or slope <= 0:
                                slope_percent = 0.1
                            else:
                                slope_percent = slope * 100
                        else:
                            slope_percent = np.nanmean(valid_slope)
                            
                        feedback.pushInfo(f"Computed mean slope from raster: {slope_percent:.4f}%")
                except Exception as e:
                    feedback.pushWarning(f"Error computing slope from raster: {str(e)}")
                    if not slope or slope <= 0:
                        slope_percent = 0.1
                    else:
                        slope_percent = slope * 100
                    feedback.pushInfo(f"Using manual slope input: {slope_percent:.4f}%")
                
                # Calculate longest flow path from vector
                longest_flow_path = self.calculate_flow_path_length(flow_path, catchment.crs, feedback)
                feedback.pushInfo(f"Computed longest flow path: {longest_flow_path:.2f} m")
            
            elif catchment_input == 1:  # Manual Parameters
                if not area or area <= 0:
                    raise QgsProcessingException("Catchment area must be >0")
                if not slope or slope <= 0:
                    slope = 0.1
                    feedback.pushWarning("Using default slope 0.1")
                slope_percent = slope * 100  # Convert to percent
                
                # Use manual flow path if provided, otherwise estimate
                if flow_path:
                    longest_flow_path = self.calculate_flow_path_length(flow_path, None, feedback)
                else:
                    area_m2 = area * 1e6
                    longest_flow_path = math.sqrt(area_m2) * 2
                    feedback.pushInfo(f"Estimated longest flow path: {longest_flow_path:.2f} m")

            if slope_percent < 0.1:
                slope_percent = 0.1
                feedback.pushWarning(f"Adjusted slope to safe minimum: {slope_percent}%")

            # === RAINFALL LOADING ===
            rainfall_df = None
            if rainfall_source == 1:  # IMD
                feedback.pushInfo(f"Downloading IMD data for {start_year}-{end_year}")
                rainfall_df = self.download_imd_data(lat, lon, start_year, end_year, imd_folder, feedback)
            elif rainfall_source == 2:  # CHIRPS
                feedback.pushInfo(f"Loading CHIRPS data from {chirps_folder}")
                rainfall_df = self.load_chirps_data(chirps_folder, lat, lon, start_year, end_year, feedback)
            
            if rainfall_df is None or rainfall_df.empty:
                raise RuntimeError("Failed to load rainfall data")
            
            # Validate rainfall data
            if rainfall_df['Rainfall (mm)'].isnull().all():
                raise RuntimeError("All rainfall values are missing")
                
            if (rainfall_df['Rainfall (mm)'] < 0).any():
                feedback.pushWarning("Negative rainfall values found. Setting to zero.")
                rainfall_df['Rainfall (mm)'] = rainfall_df['Rainfall (mm)'].clip(lower=0)
            
            # Save raw data
            raw_data_path = os.path.join(output_dir, "raw_rainfall_data.csv")
            rainfall_df.to_csv(raw_data_path)
            feedback.pushInfo(f"Raw rainfall data saved to {raw_data_path}")
            
            # === FREQUENCY ANALYSIS ===
            feedback.pushInfo("Performing frequency analysis...")
            
            # Compute annual maxima
            annual_max = rainfall_df['Rainfall (mm)'].resample('YE').max()
            
            # Handle missing values
            annual_max = annual_max.dropna()
            
            if annual_max.empty:
                raise RuntimeError("Annual maximum series is empty after dropping NaN")
                
            # Check for valid data
            if (annual_max <= 0).all():
                raise RuntimeError("All annual maximum rainfall values are <=0")
                
            feedback.pushInfo(f"Annual max rainfall stats: Min={annual_max.min():.1f}, Max={annual_max.max():.1f}, Mean={annual_max.mean():.1f}")
            feedback.pushInfo(f"Annual max values: {annual_max.values}")
            
            design_rainfalls, freq_method_name, freq_params = self.frequency_analysis(
                annual_max.values, return_periods, freq_method, feedback
            )
            
            rainfall_stats = {
                'Mean Annual Max (mm)': annual_max.mean(),
                'Std Dev (mm)': annual_max.std(),
                'Skewness': annual_max.skew(),
                'Min (mm)': annual_max.min(),
                'Max (mm)': annual_max.max(),
                'Years': len(annual_max)
            }
            
            feedback.pushInfo("\nDesign Rainfall Estimates:")
            for rp, rain in zip(return_periods, design_rainfalls):
                feedback.pushInfo(f"{rp}-year return period: {rain:.1f} mm")
            
            # === RUNOFF CALCULATIONS ===
            feedback.pushInfo("Calculating runoff...")
            
            # Calculate time to peak using SCS Lag Equation
            tp = self.calculate_time_to_peak(longest_flow_path, slope_percent, cn_value, feedback)
            feedback.pushInfo(f"Time to Peak (Tp): {tp:.2f} hr")
            
            runoff_results = []
            for rp, rain_24 in zip(return_periods, design_rainfalls):
                if runoff_method == 1:  # Rational
                    dur_hr = 1.0
                    C = 0.5
                    intensity = rain_24 / dur_hr
                    Q = 0.278 * C * area * intensity
                    method_name = "Rational"
                    col_name = f'Design Rainfall ({dur_hr:.1f}-hr) (mm)'
                    col_value = rain_24
                    
                elif runoff_method == 2:  # SCS-CN
                    runoff_depth = self.calculate_scs_cn_runoff([rain_24], cn_value)[0]
                    Q = self.calculate_peak_discharge(runoff_depth, area, tp)
                    method_name = "SCS-CN"
                    col_name = 'Runoff Depth (mm)'
                    col_value = runoff_depth
                    
                else:  # Unit Hydrograph
                    effective_rain = rain_24 * runoff_coeff_value
                    Q = self.calculate_peak_discharge(effective_rain, area, tp)
                    method_name = "Unit Hydrograph"
                    col_name = 'Effective Rainfall (mm)'
                    col_value = effective_rain
                
                # Check discharge
                if Q > 1000:
                    feedback.pushWarning(f"High discharge value ({Q:.1f} m³/s) for {rp}-year return period")
                
                runoff_results.append({
                    'Return Period (yr)': rp,
                    '24-hr Rainfall (mm)': rain_24,
                    col_name: col_value,
                    'Discharge (m³/s)': Q
                })
            
            df_out = pd.DataFrame(runoff_results)
            
            # === HYDROGRAPH GENERATION ===
            hydrograph_paths = []
            hydrograph_data = {}
            if runoff_method == 2:  # Only for SCS-CN method
                feedback.pushInfo("Generating hydrographs...")
                hydrograph_dir = os.path.join(output_dir, "hydrographs")
                os.makedirs(hydrograph_dir, exist_ok=True)
                
                # Generate unit hydrograph
                time_uh, discharge_uh = self.generate_scs_unit_hydrograph(area, tp, feedback)
                unit_hydrograph_path = os.path.join(hydrograph_dir, "unit_hydrograph.png")
                self.plot_hydrograph(time_uh, discharge_uh, "SCS Unit Hydrograph", "Time (hr)", "Discharge (m³/s)", unit_hydrograph_path)
                hydrograph_paths.append(unit_hydrograph_path)
                
                # Generate flood hydrographs for each return period
                for _, row in df_out.iterrows():
                    rp = row['Return Period (yr)']
                    runoff_depth = row['Runoff Depth (mm)']
                    
                    # Generate flood hydrograph
                    time_fh, discharge_fh = self.generate_flood_hydrograph(
                        time_uh, discharge_uh, 
                        runoff_depth, 
                        feedback
                    )
                    
                    hydrograph_data[rp] = {
                        'time': time_fh,
                        'discharge': discharge_fh
                    }
                    
                    # Plot and save
                    fh_path = os.path.join(hydrograph_dir, f"flood_hydrograph_{rp}yr.png")
                    self.plot_hydrograph(
                        time_fh, discharge_fh,
                        f"{rp}-Year Flood Hydrograph",
                        "Time (hr)", "Discharge (m³/s)",
                        fh_path
                    )
                    hydrograph_paths.append(fh_path)
            
            # === FLOOD MAPPING ===
            flood_map_paths = []
            flood_raster_paths = []
            if generate_maps and shapefile_path and dem_path:
                feedback.pushInfo("Generating flood inundation maps...")
                for _, row in df_out.iterrows():
                    rp = row['Return Period (yr)']
                    discharge = row['Discharge (m³/s)']
                    
                    flood_path, flood_raster = self.generate_flood_map(
                        discharge=discharge,
                        dem_path=dem_path,
                        catchment_path=shapefile_path,
                        output_dir=output_dir,
                        return_period=rp,
                        manning_n=manning_n_value,
                        feedback=feedback
                    )
                    
                    if flood_path and flood_raster:
                        flood_map_paths.append(flood_path)
                        flood_raster_paths.append(flood_raster)
            
            # === HEC-HMS EXPORT ===
            hec_hms_paths = []
            if export_hec_hms and runoff_method == 2:  # Only for SCS-CN
                feedback.pushInfo("Exporting HEC-HMS input files...")
                hec_dir = os.path.join(output_dir, "hec_hms")
                os.makedirs(hec_dir, exist_ok=True)
                
                # Export unit hydrograph
                uh_path = os.path.join(hec_dir, "unit_hydrograph.csv")
                pd.DataFrame({
                    'Time (hr)': time_uh,
                    'Discharge (m³/s)': discharge_uh
                }).to_csv(uh_path, index=False)
                hec_hms_paths.append(uh_path)
                
                # Export flood hydrographs
                for rp, data in hydrograph_data.items():
                    fh_path = os.path.join(hec_dir, f"flood_hydrograph_{rp}yr.csv")
                    pd.DataFrame({
                        'Time (hr)': data['time'],
                        'Discharge (m³/s)': data['discharge']
                    }).to_csv(fh_path, index=False)
                    hec_hms_paths.append(fh_path)
                
                # Export design rainfall
                for rp, rain in zip(return_periods, design_rainfalls):
                    rain_path = os.path.join(hec_dir, f"design_rainfall_{rp}yr.csv")
                    pd.DataFrame({
                        'Duration (hr)': [24],
                        'Rainfall (mm)': [rain]
                    }).to_csv(rain_path, index=False)
                    hec_hms_paths.append(rain_path)
                
                # Export parameters
                params_path = os.path.join(hec_dir, "catchment_parameters.txt")
                with open(params_path, 'w') as f:
                    f.write(f"Area (km²): {area:.2f}\n")
                    f.write(f"Curve Number: {cn_value}\n")
                    f.write(f"Time to Peak (hr): {tp:.2f}\n")
                    f.write(f"Longest Flow Path (m): {longest_flow_path:.2f}\n")
                    f.write(f"Mean Slope (%): {slope_percent:.2f}\n")
                hec_hms_paths.append(params_path)
            
            # Create summary
            summary_data = {
                'Parameter': [
                    'Catchment Area (km²)', 'Curve Number', 'Manning\'s n',
                    'Time to Peak (hr)', 'Longest Flow Path (m)', 'Mean Slope (%)',
                    'Rainfall Source', 'Frequency Method', 'Runoff Method',
                    'Rainfall Duration (hr)', 'Runoff Coefficient',
                    'Frequency Distribution Parameters', 'Annual Max Rainfall Stats'
                ],
                'Value': [
                    f"{area:.2f}", f"{cn_value}", f"{manning_n_value:.4f}",
                    f"{tp:.2f}", f"{longest_flow_path:.2f}", f"{slope_percent:.2f}",
                    'IMD' if rainfall_source == 1 else 'CHIRPS',
                    ['Gumbel', 'Log-Pearson III', 'GEV'][freq_method-1],
                    ['Rational', 'SCS-CN', 'Unit Hydrograph'][runoff_method-1],
                    "24.0",
                    f"{runoff_coeff_value:.2f}" if runoff_method == 3 else ("0.5" if runoff_method == 1 else 'N/A'),
                    str(freq_params),
                    str(rainfall_stats)
                ]
            }
            
            df_summary = pd.DataFrame(summary_data)
            
            # === SAVE RESULTS ===
            excel_path = os.path.join(output_dir, "Hydrological_Analysis_Results.xlsx")
            
            with pd.ExcelWriter(excel_path) as writer:
                df_out.to_excel(writer, sheet_name='Results', index=False)
                df_summary.to_excel(writer, sheet_name='Parameters', index=False)
                
                annual_max_df = annual_max.reset_index()
                annual_max_df.columns = ['Year', 'Annual Max Rainfall (mm)']
                annual_max_df.to_excel(writer, sheet_name='Annual Max Rainfall', index=False)
                
                if flood_map_paths:
                    flood_df = pd.DataFrame({
                        'Return Period': return_periods,
                        'Image Path': flood_map_paths,
                        'Raster Path': flood_raster_paths
                    })
                    flood_df.to_excel(writer, sheet_name='Flood Maps', index=False)
                
                if hydrograph_paths:
                    hydro_df = pd.DataFrame({
                        'Hydrograph Type': ['Unit'] + [f'{rp}-Year' for rp in return_periods],
                        'Image Path': hydrograph_paths
                    })
                    hydro_df.to_excel(writer, sheet_name='Hydrographs', index=False)
            
            # Generate plots
            self.generate_plots(output_dir, df_out, annual_max, freq_method, freq_params, feedback)
            
            # Load flood rasters in QGIS if requested
            if load_rasters and flood_raster_paths:
                feedback.pushInfo("Loading flood rasters in QGIS...")
                for raster_path in flood_raster_paths:
                    if os.path.exists(raster_path):
                        layer_name = os.path.basename(raster_path).replace('.tif', '')
                        layer = QgsRasterLayer(raster_path, layer_name)
                        if layer.isValid():
                            # Apply color ramp
                            stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
                            min_val = stats.minimumValue
                            max_val = stats.maximumValue
                            
                            # Create color ramp from blue to red
                            color_ramp = QgsColorRampShader()
                            color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
                            
                            # Define color stops
                            items = [
                                QgsColorRampShader.ColorRampItem(0.0, QColor(0, 0, 0, 0), "No Flood"),
                                QgsColorRampShader.ColorRampItem(0.01, QColor(173, 216, 230), "Shallow"),
                                QgsColorRampShader.ColorRampItem(max_val/2, QColor(0, 0, 255), "Medium"),
                                QgsColorRampShader.ColorRampItem(max_val, QColor(255, 0, 0), "Deep")
                            ]
                            color_ramp.setColorRampItemList(items)
                            
                            # Create shader
                            shader = QgsRasterShader()
                            shader.setRasterShaderFunction(color_ramp)
                            
                            # Create renderer
                            renderer = QgsSingleBandPseudoColorRenderer(
                                layer.dataProvider(), 1, shader
                            )
                            renderer.setClassificationMin(min_val)
                            renderer.setClassificationMax(max_val)
                            
                            # Apply renderer
                            layer.setRenderer(renderer)
                            layer.triggerRepaint()
                            
                            QgsProject.instance().addMapLayer(layer)
                            feedback.pushInfo(f"Loaded raster: {layer_name}")
                        else:
                            feedback.pushWarning(f"Failed to load raster: {raster_path}")
            
            # Return outputs
            flood_maps_dir = os.path.join(output_dir, "flood_maps")
            flood_rasters_dir = os.path.join(output_dir, "flood_rasters")
            hydrographs_dir = os.path.join(output_dir, "hydrographs")
            hec_hms_dir = os.path.join(output_dir, "hec_hms")
            return {
                self.OUTPUT_EXCEL: excel_path,
                self.OUTPUT_FLOOD_MAPS: flood_maps_dir,
                self.OUTPUT_FLOOD_RASTERS: flood_rasters_dir,
                self.OUTPUT_HYDROGRAPHS: hydrographs_dir,
                self.OUTPUT_HEC_HMS: hec_hms_dir
            }
            
        except Exception as e:
            feedback.reportError(f"Algorithm failed: {str(e)}")
            feedback.pushInfo(traceback.format_exc())
            raise
        finally:
            sys.stdout = original_stdout

    # ===== HELPER FUNCTIONS =====
    def parse_value_from_input(self, input_str, value_list, param_name, default, feedback):
        """
        Parse numeric value from user input (either manual number or dropdown selection)
        Supports both numeric inputs and descriptive strings
        """
        try:
            # Try to extract number from descriptive format
            if input_str in value_list:
                # User selected from dropdown
                return float(input_str.split()[0])
            
            # Try direct numeric conversion
            try:
                return float(input_str)
            except:
                pass
                
            # Extract first numeric value from string
            match = re.search(r"[-+]?\d*\.\d+|\d+", input_str)
            if match:
                return float(match.group())
                
            # If no numbers found, use default
            feedback.pushWarning(f"Couldn't parse {param_name} from '{input_str}'. Using default: {default}")
            return default
            
        except Exception as e:
            feedback.pushWarning(f"Error parsing {param_name}: {str(e)}. Using default: {default}")
            return default

    # ===== DATA PROCESSING FUNCTIONS =====
    def download_imd_data(self, lat, lon, start_year, end_year, file_dir, feedback):
        try:
            import imdlib as imd
            
            os.makedirs(file_dir, exist_ok=True)
            feedback.pushInfo(f"Downloading IMD data for {start_year}-{end_year}...")
            
            # Download IMD data
            imd.get_data('rain', start_year, end_year, fn_format='yearwise', file_dir=file_dir)
            
            actual_data_dir = os.path.join(file_dir, 'rain')
            if not os.path.exists(actual_data_dir):
                raise FileNotFoundError(f"IMD data directory not found: {actual_data_dir}")
                
            # Open IMD data
            data = imd.open_data('rain', start_year, end_year, 'yearwise', file_dir=file_dir)
            ds = data.get_xarray()
            
            # Handle missing values (-999 indicates missing data in IMD)
            ds = ds.where(ds['rain'] != -999.0)
            
            feedback.pushInfo(f"Extracting data for location: {lat:.4f}°N, {lon:.4f}°E")
            
            # Find nearest grid point with valid data
            valid_point = None
            tolerance = 1.0  # degrees tolerance
            
            for search_radius in [0, 0.1, 0.5, 1.0]:
                lat_min = lat - search_radius
                lat_max = lat + search_radius
                lon_min = lon - search_radius
                lon_max = lon + search_radius
                
                region = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                region = region.where(region['rain'].notnull(), drop=True)
                
                if region['rain'].size > 0:
                    # Calculate distances to all points in the region
                    lats = region.lat.values
                    lons = region.lon.values
                    distances = np.sqrt(
                        (lats - lat)**2 + 
                        (lons - lon)**2
                    )
                    
                    # Find the point with minimum distance
                    min_idx = np.argmin(distances)
                    lat_idx = np.where(lats == lats.flat[min_idx])[0][0]
                    lon_idx = np.where(lons == lons.flat[min_idx])[0][0]
                    
                    # Select the point
                    valid_point = region.isel(lat=lat_idx, lon=lon_idx)
                    break
            
            if valid_point is None:
                raise ValueError("No valid IMD data found near the specified location")
            
            # Extract data
            point_data = valid_point
            rainfall_df = point_data['rain'].to_dataframe().reset_index()
            rainfall_df = rainfall_df[['time', 'rain']]
            rainfall_df.columns = ['Date', 'Rainfall (mm)']
            rainfall_df.set_index('Date', inplace=True)
            rainfall_df = rainfall_df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
            
            # Validate data
            if rainfall_df['Rainfall (mm)'].isnull().all():
                raise ValueError("All IMD rainfall values are missing")
                
            feedback.pushInfo(f"Downloaded {len(rainfall_df)} days of rainfall data")
            return rainfall_df
            
        except Exception as e:
            feedback.reportError(f"IMD data processing failed: {str(e)}")
            return None

    def load_chirps_data(self, folder, lat, lon, start_year, end_year, feedback):
        try:
            import xarray as xr
            
            # Check folder exists
            if not os.path.exists(folder):
                raise FileNotFoundError(f"CHIRPS folder not found: {folder}")
            
            # Find all .nc files in the folder
            files = [f for f in os.listdir(folder) if f.endswith('.nc')]
            if not files:
                raise FileNotFoundError("No CHIRPS .nc files found")
            
            files.sort()
            feedback.pushInfo(f"Found {len(files)} CHIRPS files")
            
            # Create list to store data
            all_dates = []
            all_rain = []
            
            for year in range(start_year, end_year + 1):
                # Find file for this year
                year_files = [f for f in files if str(year) in f]
                if not year_files:
                    feedback.pushInfo(f"No CHIRPS file found for year {year}")
                    continue
                    
                file_path = os.path.join(folder, year_files[0])
                
                try:
                    # Open dataset without loading all data
                    with xr.open_dataset(file_path) as ds:
                        if 'precip' not in ds:
                            feedback.pushInfo(f"Skipping {file_path}: 'precip' variable not found")
                            continue
                        
                        # Determine longitude convention
                        lon_min = ds.longitude.values.min()
                        lon_max = ds.longitude.values.max()
                        
                        # Adjust longitude based on dataset convention
                        if lon_min >= 0 and lon_max <= 360:
                            # Dataset uses 0-360 convention
                            use_lon = lon % 360
                        elif lon_min >= -180 and lon_max <= 180:
                            # Dataset uses -180 to 180 convention
                            use_lon = lon
                        else:
                            # Unknown convention, use original longitude
                            use_lon = lon
                            feedback.pushWarning("Unknown longitude convention in CHIRPS file. Using original longitude.")
                        
                        # Find closest indices
                        lat_idx = np.abs(ds.latitude.values - lat).argmin()
                        lon_idx = np.abs(ds.longitude.values - use_lon).argmin()
                        
                        # Extract data for this point only
                        precip_data = ds['precip'][:, lat_idx, lon_idx].values
                        dates = ds['time'].values
                        
                        # Add to results
                        all_dates.extend(dates)
                        all_rain.extend(precip_data)
                        
                        # Log actual point used
                        actual_lat = ds.latitude.values[lat_idx]
                        actual_lon = ds.longitude.values[lon_idx]
                        feedback.pushInfo(f"Using CHIRPS point for {year}: lat={actual_lat}, lon={actual_lon}")
                        feedback.pushInfo(f"Loaded {year} data: {len(precip_data)} records")
                    
                except Exception as e:
                    feedback.pushInfo(f"Error processing {year}: {str(e)}")
            
            if not all_dates:
                raise ValueError("No valid CHIRPS data loaded")
                
            # Create DataFrame
            rainfall_df = pd.DataFrame({
                'Date': pd.to_datetime(all_dates),
                'Rainfall (mm)': all_rain
            }).set_index('Date')
            
            # Validate data
            if rainfall_df['Rainfall (mm)'].isnull().all():
                raise ValueError("All CHIRPS rainfall values are missing")
                
            # Filter to date range
            rainfall_df = rainfall_df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]
            
            # Set negative values to 0 and fill NaN with 0
            rainfall_df['Rainfall (mm)'] = rainfall_df['Rainfall (mm)'].fillna(0).clip(lower=0)
            
            feedback.pushInfo(f"Successfully loaded {len(rainfall_df)} CHIRPS records")
            return rainfall_df
            
        except Exception as e:
            feedback.reportError(f"CHIRPS data loading failed: {str(e)}")
            return None

    def frequency_analysis(self, data, return_periods, method, feedback):
        from scipy.stats import genextreme, gumbel_r, pearson3
        
        # Filter out zeros and negatives
        data = data[data > 0]
        if len(data) == 0:
            raise ValueError("No valid positive rainfall data for frequency analysis")
        
        if method == 1:  # Gumbel
            params = gumbel_r.fit(data)
            dist_name = "Gumbel"
            design_rainfalls = [gumbel_r.ppf(1 - 1/rp, *params) for rp in return_periods]
        elif method == 2:  # Log-Pearson III
            log_data = np.log(data)
            params = pearson3.fit(log_data)
            dist_name = "Log-Pearson III"
            design_rainfalls = [np.exp(pearson3.ppf(1 - 1/rp, *params)) for rp in return_periods]
        elif method == 3:  # GEV
            # FIX: SciPy uses opposite sign convention for shape parameter
            # Fit with floc=0 to force location parameter to be estimated
            params = genextreme.fit(data, floc=0)
            # Unpack parameters
            c, loc, scale = params
            # Convert to standard GEV parameterization
            # SciPy shape parameter c = -ξ (where ξ is standard shape parameter)
            # So for standard GEV: ξ = -c
            dist_name = "GEV"
            design_rainfalls = [genextreme.ppf(1 - 1/rp, c, loc, scale) for rp in return_periods]
        
        # Log parameters for debugging
        feedback.pushInfo(f"Fitted parameters for {dist_name}: {params}")
        feedback.pushInfo(f"Sample design rainfalls: {design_rainfalls}")
        
        # Check for unrealistic values and cap if necessary
        max_reasonable_rainfall = 5000  # 5000 mm/day is world record * 3
        if max(design_rainfalls) > max_reasonable_rainfall:
            feedback.pushWarning("Unrealistically high design rainfall detected. Recomputing with Gumbel method.")
            params = gumbel_r.fit(data)
            design_rainfalls = [gumbel_r.ppf(1 - 1/rp, *params) for rp in return_periods]
            dist_name = "Gumbel (fallback)"
            # Cap individual values that are still too high
            design_rainfalls = [min(x, max_reasonable_rainfall) for x in design_rainfalls]
            feedback.pushInfo(f"Recalculated with Gumbel: {design_rainfalls}")
        
        return design_rainfalls, dist_name, params

    def calculate_scs_cn_runoff(self, rainfall, cn):
        S = (25400 / cn) - 254
        Ia = 0.2 * S
        runoff = np.zeros_like(rainfall)
        for i, r in enumerate(rainfall):
            if r <= Ia:
                runoff[i] = 0
            else:
                runoff[i] = (r - Ia)**2 / (r - Ia + S)
        return runoff

    def calculate_peak_discharge(self, runoff_depth, area, tp):
        return (0.208 * area * runoff_depth) / tp

    def calculate_time_to_peak(self, longest_flow_path, slope_percent, cn, feedback):
        """
        Calculate time to peak using SCS Lag Equation
        Tp = (L^0.8 * (S + 1)^0.7) / (1900 * slope^0.5)
        Where:
            L = longest flow path (feet)
            S = potential maximum retention (inches) = (1000/CN) - 10
            slope = average watershed slope (%)
        """
        # Convert longest flow path from meters to feet
        L_ft = longest_flow_path * 3.28084
        
        # Calculate S (potential maximum retention)
        S = (1000.0 / cn) - 10.0
        
        # Calculate Tp in hours
        numerator = (L_ft ** 0.8) * ((S + 1) ** 0.7)
        denominator = 1900.0 * (slope_percent ** 0.5)
        Tp = numerator / denominator
        
        feedback.pushInfo(f"SCS Lag Equation: L={longest_flow_path:.1f}m ({L_ft:.1f}ft), S={S:.1f}in, slope={slope_percent:.1f}% -> Tp={Tp:.2f} hr")
        return Tp

    def calculate_flow_path_length(self, flow_path, target_crs, feedback):
        """Calculate length of longest flow path from vector"""
        try:
            import geopandas as gpd
            
            flow_path_gdf = gpd.read_file(flow_path)
            
            # Reproject to target CRS if needed
            if target_crs and flow_path_gdf.crs != target_crs:
                flow_path_gdf = flow_path_gdf.to_crs(target_crs)
            
            # Calculate length of all features
            lengths = flow_path_gdf.geometry.length
            longest_length = max(lengths)  # in meters
            
            if longest_length <= 0:
                raise ValueError("Invalid flow path length (<=0)")
                
            feedback.pushInfo(f"Longest flow path: {longest_length:.2f} m")
            return longest_length
            
        except Exception as e:
            feedback.reportError(f"Error calculating flow path: {str(e)}")
            return 0

    def generate_flood_map(self, discharge, dem_path, catchment_path, output_dir, return_period, manning_n, feedback):
        import geopandas as gpd
        import rasterio
        from rasterio.mask import mask
        from rasterio import features
        from shapely.geometry import mapping
        from rasterio.plot import plotting_extent
        
        flood_maps_dir = os.path.join(output_dir, "flood_maps")
        flood_rasters_dir = os.path.join(output_dir, "flood_rasters")
        os.makedirs(flood_maps_dir, exist_ok=True)
        os.makedirs(flood_rasters_dir, exist_ok=True)
        
        flood_image = os.path.join(flood_maps_dir, f"flood_map_{return_period}yr.png")
        flood_raster = os.path.join(flood_rasters_dir, f"flood_depth_{return_period}yr.tif")
        
        try:
            # Force garbage collection to release file handles
            gc.collect()
            
            catchment = gpd.read_file(catchment_path)
            
            with rasterio.open(dem_path) as src:
                catchment = catchment.to_crs(src.crs)
                geoms = [mapping(geom) for geom in catchment.geometry]
                out_image, out_transform = mask(src, geoms, crop=True, nodata=np.nan)
                dem_data = out_image[0]
                
                # Create catchment mask
                catchment_mask = features.geometry_mask(
                    catchment.geometry,
                    transform=out_transform,
                    out_shape=src.shape,
                    invert=True
                )
                
                # Save extent for plotting
                flood_extent = plotting_extent(dem_data, out_transform)
                profile = src.profile.copy()
            
            # Calculate water depth using Manning's equation
            try:
                # Estimate hydraulic radius
                hydraulic_radius = np.nanmean(dem_data) * 0.7
                slope_val = 0.01  # Approximate slope
                
                # Calculate water depth using Manning's equation
                numerator = discharge * manning_n
                denominator = np.sqrt(slope_val) * hydraulic_radius**(2/3)
                water_depth = (numerator / denominator) ** (3/5)
                feedback.pushInfo(f"Calculated water depth using Manning's equation: {water_depth:.2f} m (n={manning_n})")
            except:
                # Fallback method
                water_depth = max(0.1, (discharge / 100) * 0.5)
                feedback.pushWarning("Using fallback water depth calculation")
            
            # Calculate water surface
            min_elev = np.nanmin(dem_data)
            water_surface = min_elev + water_depth
            
            # Calculate flood extent and depth
            flood_mask = (dem_data < water_surface)
            flood_depth = np.where(flood_mask, water_surface - dem_data, 0)
            
            # Apply catchment mask
            flood_depth = np.where(catchment_mask, flood_depth, 0)
            
            # Save as GeoTIFF using the saved profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=0  # Set nodata to 0 for better symbology
            )
            
            with rasterio.open(flood_raster, 'w', **profile) as dst:
                dst.write(flood_depth.astype(rasterio.float32), 1)
            
            # Create custom colormap: blue (low) to red (high)
            colors = [(0, 0, 1, 0.7), (1, 0, 0, 0.9)]  # Blue to Red with alpha
            cmap = LinearSegmentedColormap.from_list("flood_cmap", colors, N=256)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot DEM background
            dem_plot = ax.imshow(dem_data, extent=flood_extent, cmap='terrain', alpha=0.6)
            
            # Add flood overlay with new colormap
            masked_flood = np.ma.masked_where((flood_depth < 0.01) | np.isnan(flood_depth), flood_depth)
            flood_layer = ax.imshow(masked_flood, extent=flood_extent, cmap=cmap, vmin=0, vmax=np.nanmax(flood_depth), alpha=0.8)
            
            # Add colorbar
            cbar = plt.colorbar(flood_layer, ax=ax)
            cbar.set_label('Flood Depth (m)')
            
            # Add title and save
            ax.set_title(f"Flood Extent for {return_period}-Year Return Period\nDischarge: {discharge:.1f} m³/s | Depth: {water_depth:.1f} m | Manning's n: {manning_n}")
            plt.savefig(flood_image, dpi=300, bbox_inches='tight')
            plt.close()
            
            feedback.pushInfo(f"Generated flood map for {return_period}-year return period")
            return flood_image, flood_raster
            
        except Exception as e:
            feedback.reportError(f"Flood map generation failed: {str(e)}")
            return None, None

    def generate_scs_unit_hydrograph(self, area_km2, tp_hr, feedback):
        """
        Generate SCS triangular unit hydrograph
        :param area_km2: Catchment area in km²
        :param tp_hr: Time to peak in hours
        :return: time array (hr), discharge array (m³/s)
        """
        try:
            # Time base of hydrograph (hr)
            tb = 8/3 * tp_hr
            
            # Peak discharge for 1 cm of runoff
            Qp = (2.08 * area_km2) / tp_hr
            
            # Create time array
            time_step = 0.1  # hours
            time = np.arange(0, tb + time_step, time_step)
            discharge = np.zeros_like(time)
            
            # Rising limb
            rising_mask = time <= tp_hr
            discharge[rising_mask] = Qp * (time[rising_mask] / tp_hr)
            
            # Falling limb
            falling_mask = (time > tp_hr) & (time <= tb)
            discharge[falling_mask] = Qp * (tb - time[falling_mask]) / (tb - tp_hr)
            
            feedback.pushInfo(f"Generated SCS unit hydrograph: Tp={tp_hr:.2f} hr, Tb={tb:.2f} hr, Qp={Qp:.2f} m³/s")
            return time, discharge
            
        except Exception as e:
            feedback.reportError(f"Unit hydrograph generation failed: {str(e)}")
            return np.array([0, 1, 2]), np.array([0, 0, 0])

    def generate_flood_hydrograph(self, time_uh, discharge_uh, runoff_depth_cm, feedback):
        """
        Generate flood hydrograph by scaling unit hydrograph
        :param time_uh: Unit hydrograph time array (hr)
        :param discharge_uh: Unit hydrograph discharge array (m³/s)
        :param runoff_depth_cm: Total runoff depth in cm
        :return: time array (hr), discharge array (m³/s)
        """
        try:
            # Convert runoff depth from mm to cm
            runoff_depth_cm = runoff_depth_cm / 10
            
            # Scale unit hydrograph by runoff depth
            scaled_discharge = discharge_uh * runoff_depth_cm
            
            feedback.pushInfo(f"Scaled unit hydrograph by runoff depth: {runoff_depth_cm:.1f} cm")
            return time_uh, scaled_discharge
            
        except Exception as e:
            feedback.reportError(f"Flood hydrograph generation failed: {str(e)}")
            return np.array([0, 1, 2]), np.array([0, 0, 0])

    def plot_hydrograph(self, time, discharge, title, xlabel, ylabel, output_path):
        """Plot and save hydrograph"""
        plt.figure(figsize=(10, 6))
        plt.plot(time, discharge, 'b-', linewidth=2)
        plt.fill_between(time, discharge, alpha=0.3, color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_plots(self, output_dir, df_out, annual_max, freq_method, freq_params, feedback):
        # Results plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        ax1.plot(df_out['Return Period (yr)'], df_out['24-hr Rainfall (mm)'], 'o-', color='royalblue', linewidth=2)
        ax1.set_title(f"Design Rainfall (24-hr Duration)")
        ax1.set_xlabel("Return Period (years)")
        ax1.set_ylabel("Rainfall Depth (mm)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        ax2.plot(df_out['Return Period (yr)'], df_out['Discharge (m³/s)'], 's-', color='crimson', linewidth=2)
        ax2.set_title("Design Discharge")
        ax2.set_xlabel("Return Period (years)")
        ax2.set_ylabel("Peak Discharge (m³/s)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        results_plot_path = os.path.join(output_dir, "Results_Plot.png")
        plt.savefig(results_plot_path, dpi=300)
        plt.close()
        
        # Frequency distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(annual_max.values, bins=15, density=True, alpha=0.7, label='Annual Max Rainfall')
        x = np.linspace(min(annual_max), max(annual_max)*1.5, 100)
        
        if freq_method == 1:  # Gumbel
            from scipy.stats import gumbel_r
            dist = gumbel_r(*freq_params)
        elif freq_method == 2:  # Log-Pearson III
            from scipy.stats import pearson3
            dist = pearson3(*freq_params)
        else:  # GEV
            from scipy.stats import genextreme
            dist = genextreme(*freq_params)
        
        plt.plot(x, dist.pdf(x), 'r-', lw=2, label=f'Fit')
        plt.title('Frequency Distribution Fit')
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        freq_plot_path = os.path.join(output_dir, "Frequency_Distribution.png")
        plt.savefig(freq_plot_path, dpi=300)
        plt.close()
        
        feedback.pushInfo(f"Charts saved to output directory")

    def name(self):
        return 'rain2flood'

    def displayName(self):
        return 'Rain to Flood Analysis'

    def group(self):
        return 'Hydrology'

    def groupId(self):
        return 'hydrology'

    def createInstance(self):
        return Rain2FloodAlgorithm()

class Rain2FloodProvider(QgsProcessingProvider):
    def __init__(self):
        super().__init__()
    
    def loadAlgorithms(self):
        # Check dependencies before loading algorithm
        missing = DependencyManager.check_dependencies()
        if missing:
            # Show install instructions
            DependencyManager.show_install_dialog()
            return
        self.addAlgorithm(Rain2FloodAlgorithm())
    
    def id(self):
        return 'rain2flood'
    
    def name(self):
        return 'Rain to Flood Toolkit'
    
    def icon(self):
        return QIcon()

def provider():
    return Rain2FloodProvider()