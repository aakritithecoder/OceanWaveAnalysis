# Ocean Wave Height Analysis

A scientific computing project demonstrating wave height simulation, analysis, and visualization using Python.

**Author:** Aakriti  
**GitHub:** [aakritithecoder](https://github.com/aakritithecoder)

## Overview

This project generates synthetic ocean wave data, performs statistical analysis, and creates visualizations. It demonstrates:
- Multi-dimensional data handling with xarray
- Numerical computing with numpy
- Statistical analysis
- Data visualization with matplotlib
- Unit testing with pytest

## Features

- **Data Generation**: Creates 30 days of synthetic wave height and direction data on a 9×9 spatial grid with 3-hour temporal resolution
- **Statistical Analysis**: Computes mean, median, standard deviation, percentiles, and other statistics
- **Visualizations**:
  - Time series plot of average wave height
  - Histogram showing wave height distribution
  - Spatial contour map at a specific time
- **NetCDF Support**: Save and load data in NetCDF format
- **Unit Tests**: 12 pytest tests covering all functionality

## Requirements

```bash
pip install numpy xarray pandas matplotlib pytest