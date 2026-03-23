"""
Unit tests for Ocean Wave Height Analysis
"""

import pytest
import numpy as np
import xarray as xr
from wave_analysis import WaveAnalyzer


class TestWaveAnalyzer:
    """Test suite for WaveAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Fixture providing a WaveAnalyzer instance"""
        return WaveAnalyzer(lat_min=10.0, lat_max=10.8,
                           lon_min=20.0, lon_max=20.8)
    
    @pytest.fixture
    def test_data(self, analyzer):
        """Fixture providing test wave data"""
        return analyzer.generate_wave_data(days=7, timestep_hours=3, seed=42)
    
    def test_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer.lat_min == 10.0
        assert analyzer.lat_max == 10.8
        assert analyzer.lon_min == 20.0
        assert analyzer.lon_max == 20.8
        assert len(analyzer.latitudes) == 9  # 10.0 to 10.8 step 0.1
        assert len(analyzer.longitudes) == 9
    
    def test_generate_wave_data_shape(self, test_data):
        """Test that generated data has correct dimensions"""
        assert 'wave_height' in test_data.data_vars
        assert 'wave_direction' in test_data.data_vars
        assert 'time' in test_data.dims
        assert 'latitude' in test_data.dims
        assert 'longitude' in test_data.dims
    
    def test_generate_wave_data_time_steps(self, test_data):
        """Test that number of time steps is correct (7 days, 3h timestep = 56 steps)"""
        # 7 days × 24 hours ÷ 3 hours = 56 timesteps
        expected_steps = 56
        assert len(test_data.time) == expected_steps
    
    def test_wave_height_range(self, test_data):
        """Test that wave heights are within realistic range (0.5-8m)"""
        heights = test_data['wave_height'].values
        assert np.all(heights >= 0.5)
        assert np.all(heights <= 8.0)
    
    def test_wave_direction_range(self, test_data):
        """Test that wave directions are within 0-360 degrees"""
        directions = test_data['wave_direction'].values
        assert np.all(directions >= 0)
        assert np.all(directions < 360)
    
    def test_compute_statistics_returns_dict(self, analyzer, test_data):
        """Test that statistics function returns a dictionary"""
        stats = analyzer.compute_statistics(test_data)
        assert isinstance(stats, dict)
        assert 'mean_height' in stats
        assert 'std_height' in stats
        assert 'max_height' in stats
    
    def test_compute_statistics_values_positive(self, analyzer, test_data):
        """Test that all statistics are positive"""
        stats = analyzer.compute_statistics(test_data)
        for key, value in stats.items():
            if 'direction' not in key:
                assert value > 0, f"{key} should be positive, got {value}"
    
    def test_compute_time_series_returns_series(self, analyzer, test_data):
        """Test that time series function returns pandas Series"""
        daily_means = analyzer.compute_time_series(test_data)
        import pandas as pd
        assert isinstance(daily_means, pd.Series)
        assert len(daily_means) == 7  # 7 days
    
    def test_reproducibility_with_seed(self):
        """Test that using same seed produces identical data"""
        analyzer1 = WaveAnalyzer()
        data1 = analyzer1.generate_wave_data(days=7, seed=42)
        
        analyzer2 = WaveAnalyzer()
        data2 = analyzer2.generate_wave_data(days=7, seed=42)
        
        assert np.all(data1['wave_height'].values == data2['wave_height'].values)
    
    def test_save_and_load_netcdf(self, analyzer, test_data, tmp_path):
        """Test saving and loading NetCDF file"""
        filepath = tmp_path / "test_data.nc"
        analyzer.save_to_netcdf(test_data, filepath)
        loaded_data = analyzer.load_from_netcdf(filepath)
        
        assert loaded_data.identical(test_data)
    
    def test_statistical_reproducibility(self, analyzer, test_data):
        """Test that statistical calculations are reproducible"""
        stats1 = analyzer.compute_statistics(test_data)
        
        # Generate data again with same seed
        test_data2 = analyzer.generate_wave_data(days=7, timestep_hours=3, seed=42)
        stats2 = analyzer.compute_statistics(test_data2)
        
        # Both runs should produce identical statistics
        assert stats1['mean_height'] == stats2['mean_height']
        assert stats1['std_height'] == stats2['std_height']
        assert stats1['max_height'] == stats2['max_height']
    
    def test_statistical_reasonable_values(self, analyzer, test_data):
        """Test that statistical values are within reasonable ranges"""
        stats = analyzer.compute_statistics(test_data)
        
        # Wave heights should be between 0.5 and 8 meters
        assert 0.5 < stats['mean_height'] < 8
        assert 0 < stats['std_height'] < 3
        assert 0.5 < stats['min_height'] < 3
        assert 2 < stats['max_height'] < 9
        assert 0 < stats['percentile_90'] < 8