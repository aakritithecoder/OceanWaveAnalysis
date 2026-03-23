"""
Ocean Wave Height Analysis
A scientific computing project for GSoC application
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class WaveAnalyzer:
    """Class to generate and analyze wave height data"""
    
    def __init__(self, lat_min=10.0, lat_max=10.8, lon_min=20.0, lon_max=20.8, resolution=0.1):
        """
        Initialize the analyzer with geographic bounds
        
        Parameters:
        - lat_min, lat_max: latitude range
        - lon_min, lon_max: longitude range
        - resolution: grid spacing in degrees
        """
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.resolution = resolution
        
        # Create coordinate arrays
        self.latitudes = np.arange(lat_min, lat_max + resolution/2, resolution)
        self.longitudes = np.arange(lon_min, lon_max + resolution/2, resolution)
        
        print(f"Created grid: {len(self.latitudes)} lat × {len(self.longitudes)} lon")
    
    def generate_wave_data(self, days=30, timestep_hours=3, seed=42):
        """
        Generate synthetic wave height data
        
        Parameters:
        - days: number of days to generate
        - timestep_hours: time resolution in hours
        - seed: random seed for reproducibility
        
        Returns:
        - xarray.Dataset with wave height and direction
        """
        np.random.seed(seed)
        
        # Create time array
        start_time = datetime(2026, 3, 4, 0, 0, 0)
        time_steps = int(days * 24 / timestep_hours)
        times = [start_time + timedelta(hours=i * timestep_hours) 
                 for i in range(time_steps)]
        
        print(f"Generating {len(times)} time steps over {days} days")
        
        # Create empty grids
        n_time = len(times)
        n_lat = len(self.latitudes)
        n_lon = len(self.longitudes)
        
        wave_height = np.zeros((n_time, n_lat, n_lon))
        wave_direction = np.zeros((n_time, n_lat, n_lon))
        
        # Generate data with realistic patterns
        for t, time in enumerate(times):
            # Time factor (diurnal variation)
            hour_of_day = time.hour
            diurnal = 0.5 * np.sin(hour_of_day * np.pi / 12)
            
            for i, lat in enumerate(self.latitudes):
                for j, lon in enumerate(self.longitudes):
                    # Base height: 2m
                    base_height = 2.0
                    
                    # Latitude effect (higher waves at higher latitudes)
                    lat_effect = (lat - self.lat_min) / (self.lat_max - self.lat_min) * 1.5
                    
                    # Time variation (storm passing through)
                    storm_effect = 1.0 + np.sin(t * 2 * np.pi / 20) * 0.5
                    
                    # Random noise
                    noise = np.random.normal(0, 0.2)
                    
                    # Final wave height
                    height = base_height + lat_effect + diurnal + storm_effect * 0.5 + noise
                    height = max(0.5, min(8.0, height))  # Clamp between 0.5-8m
                    
                    # Wave direction (from wind direction)
                    direction = 180 + 30 * np.sin(t * 2 * np.pi / 30) + (lat - self.lat_min) * 20
                    direction = direction % 360
                    
                    wave_height[t, i, j] = height
                    wave_direction[t, i, j] = direction
        
        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                'wave_height': (['time', 'latitude', 'longitude'], wave_height),
                'wave_direction': (['time', 'latitude', 'longitude'], wave_direction)
            },
            coords={
                'time': times,
                'latitude': self.latitudes,
                'longitude': self.longitudes
            }
        )
        
        # Add units
        ds['wave_height'].attrs['units'] = 'm'
        ds['wave_direction'].attrs['units'] = 'degrees'
        
        return ds
    
    def compute_statistics(self, ds):
        """
        Compute statistical summaries of wave data
        
        Returns:
        - dict with statistics
        """
        stats = {
            'mean_height': float(ds['wave_height'].mean()),
            'median_height': float(ds['wave_height'].median()),
            'std_height': float(ds['wave_height'].std()),
            'max_height': float(ds['wave_height'].max()),
            'min_height': float(ds['wave_height'].min()),
            'mean_direction': float(ds['wave_direction'].mean()),
            'percentile_90': float(ds['wave_height'].quantile(0.9))
        }
        return stats
    
    def compute_time_series(self, ds):
        """
        Compute daily averages for time series plotting
        """
        # Convert time to pandas datetime for grouping
        df = ds.to_dataframe().reset_index()
        df['date'] = pd.to_datetime(df['time']).dt.date
        
        # Group by date and compute daily means
        daily_means = df.groupby('date')['wave_height'].mean()
        
        return daily_means
    
    def plot_wave_height_time_series(self, ds, save_path=None):
        """
        Create time series plot of average wave height
        """
        daily_means = self.compute_time_series(ds)
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_means.index, daily_means.values, 'b-', linewidth=2)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Wave Height (m)', fontsize=12)
        plt.title('Average Wave Height Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved time series plot to {save_path}")
        
        plt.close()
    
    def plot_wave_height_histogram(self, ds, save_path=None):
        """
        Create histogram of wave heights
        """
        heights = ds['wave_height'].values.flatten()
        
        plt.figure(figsize=(10, 6))
        plt.hist(heights, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Wave Height (m)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Wave Heights', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved histogram to {save_path}")
        
        plt.close()
    
    def plot_spatial_map(self, ds, time_index=0, save_path=None):
        """
        Create spatial map of wave heights at a specific time
        """
        wave_height = ds['wave_height'].isel(time=time_index)
        
        plt.figure(figsize=(10, 8))
        im = plt.contourf(ds['longitude'], ds['latitude'], wave_height, 
                          levels=20, cmap='Blues')
        plt.colorbar(im, label='Wave Height (m)')
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.title(f'Wave Height Map at {ds.time.values[time_index]}', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved spatial map to {save_path}")
        
        plt.close()
    
    def save_to_netcdf(self, ds, filepath):
        """
        Save dataset to NetCDF file
        """
        ds.to_netcdf(filepath)
        print(f"Saved data to {filepath}")
    
    def load_from_netcdf(self, filepath):
        """
        Load dataset from NetCDF file
        """
        return xr.open_dataset(filepath)


def main():
    """Example usage of the WaveAnalyzer"""
    
    print("=" * 50)
    print("Ocean Wave Height Analysis")
    print("=" * 50)
    
    # Create analyzer
    analyzer = WaveAnalyzer(lat_min=10.0, lat_max=10.8, 
                            lon_min=20.0, lon_max=20.8)
    
    # Generate data
    print("\n1. Generating wave data...")
    ds = analyzer.generate_wave_data(days=30, timestep_hours=3)
    
    # Compute statistics
    print("\n2. Computing statistics...")
    stats = analyzer.compute_statistics(ds)
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}")
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    analyzer.plot_wave_height_time_series(ds, save_path="figures/time_series.png")
    analyzer.plot_wave_height_histogram(ds, save_path="figures/histogram.png")
    analyzer.plot_spatial_map(ds, time_index=0, save_path="figures/spatial_map.png")
    
    # Save data
    print("\n4. Saving data...")
    analyzer.save_to_netcdf(ds, "data/wave_data.nc")
    
    print("\n✅ Analysis complete!")
    print("   Figures saved to 'figures/' directory")
    print("   Data saved to 'data/wave_data.nc'")


if __name__ == "__main__":
    main()