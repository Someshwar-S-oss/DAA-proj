"""
Anomaly detection module for identifying outlier listings.
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Detects anomalous Airbnb listings using multiple statistical methods.
    """
    
    def __init__(self, data, metrics):
        """
        Initialize the anomaly detector.
        
        Args:
            data (pd.DataFrame): Airbnb listings data
            metrics (dict): Dictionary of metric arrays (cell_area, nearest_neighbor, etc.)
        """
        self.data = data
        self.metrics = metrics
        self.anomaly_scores = None
        
    def zscore_method(self, values, threshold=3):
        """
        Detect anomalies using Z-score method.
        
        Args:
            values (np.ndarray): Array of values to analyze
            threshold (float): Z-score threshold for anomalies
            
        Returns:
            tuple: (anomaly mask, z-scores)
        """
        # Remove infinite values
        finite_mask = ~np.isinf(values)
        z_scores = np.zeros_like(values, dtype=float)
        
        if finite_mask.sum() > 0:
            z_scores[finite_mask] = np.abs(stats.zscore(values[finite_mask], nan_policy='omit'))
        
        anomalies = z_scores > threshold
        return anomalies, z_scores
    
    def iqr_method(self, values):
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            values (np.ndarray): Array of values to analyze
            
        Returns:
            np.ndarray: Boolean array indicating anomalies
        """
        # Filter out infinite values
        finite_values = values[~np.isinf(values)]
        
        if len(finite_values) == 0:
            return np.zeros_like(values, dtype=bool)
        
        q1 = np.percentile(finite_values, 25)
        q3 = np.percentile(finite_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = (values < lower_bound) | (values > upper_bound)
        return anomalies
    
    def percentile_method(self, values, percentile=95):
        """
        Detect top percentile as anomalies.
        
        Args:
            values (np.ndarray): Array of values to analyze
            percentile (float): Percentile threshold (default 95)
            
        Returns:
            np.ndarray: Boolean array indicating anomalies
        """
        finite_values = values[~np.isinf(values)]
        
        if len(finite_values) == 0:
            return np.zeros_like(values, dtype=bool)
        
        threshold = np.percentile(finite_values, percentile)
        anomalies = values > threshold
        return anomalies
    
    def isolation_forest_method(self, features, contamination=0.1):
        """
        Use Isolation Forest for multi-dimensional anomaly detection.
        
        Args:
            features (np.ndarray): 2D array of features
            contamination (float): Expected proportion of anomalies
            
        Returns:
            tuple: (anomaly mask, anomaly scores)
        """
        # Replace infinite values with large finite numbers
        features_clean = features.copy()
        for col in range(features_clean.shape[1]):
            finite_max = features_clean[~np.isinf(features_clean[:, col]), col].max()
            features_clean[np.isinf(features_clean[:, col]), col] = finite_max * 2
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = iso_forest.fit_predict(features_scaled)
        anomalies = predictions == -1
        scores = iso_forest.score_samples(features_scaled)
        
        # Convert scores to 0-1 range (higher = more anomalous)
        scores_normalized = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        print(f"Isolation Forest detected {anomalies.sum()} anomalies ({anomalies.sum()/len(anomalies)*100:.2f}%)")
        
        return anomalies, scores_normalized
    
    def composite_score(self, weights=None):
        """
        Calculate weighted composite anomaly score from multiple metrics.
        
        Args:
            weights (dict, optional): Weights for each metric
            
        Returns:
            np.ndarray: Composite anomaly scores (0-1 range)
        """
        if weights is None:
            weights = {
                'cell_area': 0.4,
                'nearest_neighbor': 0.3,
                'density': 0.2,
                'price': 0.1
            }
        
        # Normalize each metric to 0-1
        normalized = {}
        
        for key, values in self.metrics.items():
            if key not in weights:
                continue
                
            # Handle infinite values
            finite_mask = ~np.isinf(values)
            
            if finite_mask.sum() == 0:
                normalized[key] = np.zeros_like(values)
                continue
            
            valid_values = values[finite_mask]
            min_val = valid_values.min()
            max_val = valid_values.max()
            
            # Normalize to 0-1
            norm_values = np.zeros_like(values, dtype=float)
            if max_val > min_val:
                norm_values[finite_mask] = (values[finite_mask] - min_val) / (max_val - min_val)
                norm_values[~finite_mask] = 1.0  # Infinite values get max score
            
            # For density, invert (low density = high anomaly)
            if key == 'density':
                norm_values = 1 - norm_values
            
            normalized[key] = norm_values
        
        # Calculate weighted sum
        score = np.zeros(len(self.metrics[list(self.metrics.keys())[0]]))
        total_weight = 0
        
        for key in weights.keys():
            if key in normalized:
                score += normalized[key] * weights[key]
                total_weight += weights[key]
        
        # Normalize to 0-1 range
        if total_weight > 0:
            score = score / total_weight
        
        self.anomaly_scores = score
        
        print(f"Calculated composite anomaly scores")
        print(f"  Mean score: {score.mean():.3f}")
        print(f"  Max score: {score.max():.3f}")
        
        return score
    
    def classify_anomalies(self, threshold=0.7):
        """
        Classify listings into anomaly categories.
        
        Args:
            threshold (float): Score threshold for anomaly classification
            
        Returns:
            np.ndarray: Array of category labels
        """
        if self.anomaly_scores is None:
            raise ValueError("Anomaly scores not calculated. Call composite_score() first.")
        
        categories = []
        
        for idx, score in enumerate(self.anomaly_scores):
            if score < threshold:
                categories.append('Normal')
            else:
                # Determine specific category based on metrics
                area = self.metrics.get('cell_area', [0])[idx]
                price = self.metrics.get('price', [0])[idx]
                density = self.metrics.get('density', [0])[idx]
                
                # Calculate percentiles
                area_percentile = 50
                price_percentile = 50
                
                if 'cell_area' in self.metrics:
                    finite_areas = self.metrics['cell_area'][~np.isinf(self.metrics['cell_area'])]
                    if len(finite_areas) > 0 and not np.isinf(area):
                        area_percentile = stats.percentileofscore(finite_areas, area)
                
                if 'price' in self.metrics:
                    price_percentile = stats.percentileofscore(self.metrics['price'], price)
                
                # Classify based on dominant factor
                if area_percentile > 90 or np.isinf(area):
                    categories.append('Isolated')
                elif price_percentile > 90:
                    categories.append('Price Anomaly')
                elif density < np.percentile(self.metrics['density'], 10):
                    categories.append('Low Density')
                else:
                    categories.append('Multi-factor Anomaly')
        
        categories = np.array(categories)
        
        # Print statistics
        unique, counts = np.unique(categories, return_counts=True)
        print(f"\nAnomaly Classification (threshold={threshold}):")
        for cat, count in zip(unique, counts):
            print(f"  {cat}: {count} ({count/len(categories)*100:.2f}%)")
        
        return categories
    
    def get_top_anomalies(self, n=20):
        """
        Get the top n most anomalous listings.
        
        Args:
            n (int): Number of top anomalies to return
            
        Returns:
            pd.DataFrame: Top anomalous listings with scores
        """
        if self.anomaly_scores is None:
            raise ValueError("Anomaly scores not calculated. Call composite_score() first.")
        
        # Add scores to dataframe
        df_with_scores = self.data.copy()
        df_with_scores['anomaly_score'] = self.anomaly_scores
        
        # Add relevant metrics
        for key, values in self.metrics.items():
            df_with_scores[f'metric_{key}'] = values
        
        # Sort by anomaly score
        top_anomalies = df_with_scores.nlargest(n, 'anomaly_score')
        
        return top_anomalies
