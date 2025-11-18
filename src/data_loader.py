"""
Data loading and preprocessing module for Airbnb listings.
"""

import pandas as pd
import numpy as np


class AirbnbDataLoader:
    """
    Handles loading, cleaning, and preprocessing of Airbnb listings data.
    """
    
    def __init__(self, filepath):
        """
        Initialize the data loader.
        
        Args:
            filepath (str): Path to the CSV file containing Airbnb listings
        """
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """
        Load CSV file into pandas DataFrame.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} listings")
        return self.df
    
    def clean_data(self):
        """
        Clean and validate data by:
        - Removing missing coordinates
        - Removing invalid coordinates (0,0 or out of bounds)
        - Handling price outliers
        - Ensuring valid data types
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        initial_count = len(self.df)
        
        # Remove missing coordinates
        self.df = self.df.dropna(subset=['latitude', 'longitude'])
        
        # Remove invalid coordinates (0,0)
        self.df = self.df[
            (self.df['latitude'] != 0) & 
            (self.df['longitude'] != 0)
        ]
        
        # Handle missing or zero prices
        self.df = self.df[self.df['price'].notna()]
        self.df = self.df[self.df['price'] > 0]
        
        # Remove extreme price outliers (optional - keeps 1st to 99th percentile)
        q1 = self.df['price'].quantile(0.01)
        q99 = self.df['price'].quantile(0.99)
        self.df = self.df[
            (self.df['price'] >= q1) & 
            (self.df['price'] <= q99)
        ]
        
        # Reset index
        self.df = self.df.reset_index(drop=True)
        
        removed_count = initial_count - len(self.df)
        print(f"Cleaned data: {len(self.df)} listings (removed {removed_count})")
        return self.df
    
    def get_coordinates(self):
        """
        Extract coordinate arrays for Voronoi generation.
        
        Returns:
            np.ndarray: Array of [longitude, latitude] pairs
        """
        points = self.df[['longitude', 'latitude']].values
        return points
    
    def get_summary(self):
        """
        Generate statistical summary of the dataset.
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_listings': len(self.df),
            'price_range': (self.df['price'].min(), self.df['price'].max()),
            'avg_price': self.df['price'].mean(),
            'median_price': self.df['price'].median(),
            'neighborhoods': self.df['neighbourhood'].nunique() if 'neighbourhood' in self.df.columns else 'N/A',
            'room_types': self.df['room_type'].value_counts().to_dict() if 'room_type' in self.df.columns else 'N/A',
            'lat_range': (self.df['latitude'].min(), self.df['latitude'].max()),
            'lon_range': (self.df['longitude'].min(), self.df['longitude'].max())
        }
        return summary
    
    def get_bounding_box(self, padding=0.01):
        """
        Calculate bounding box for the listings area.
        
        Args:
            padding (float): Padding to add around the bounding box
            
        Returns:
            list: Bounding box as [(min_lon, min_lat), (max_lon, min_lat), 
                                   (max_lon, max_lat), (min_lon, max_lat)]
        """
        min_lon = self.df['longitude'].min() - padding
        max_lon = self.df['longitude'].max() + padding
        min_lat = self.df['latitude'].min() - padding
        max_lat = self.df['latitude'].max() + padding
        
        bbox = [
            (min_lon, min_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
            (min_lon, max_lat)
        ]
        return bbox
