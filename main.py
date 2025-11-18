"""
Main pipeline for Airbnb Voronoi Anomaly Detection.

This script orchestrates the complete analysis workflow:
1. Load and clean data
2. Generate Voronoi diagram
3. Calculate spatial metrics
4. Detect anomalies
5. Create visualizations
6. Generate reports
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import AirbnbDataLoader
from src.voronoi_generator import VoronoiAnalyzer
from src.anomaly_detector import AnomalyDetector
from src.visualizer import VoronoiVisualizer


def create_directories():
    """Create output directories if they don't exist."""
    directories = [
        'output',
        'output/figures',
        'output/reports',
        'output/maps'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_menu():
    """Display the main menu."""
    print("\n" + "=" * 70)
    print("  MAIN MENU")
    print("=" * 70)
    print("  1. Load and analyze data")
    print("  2. View dataset summary")
    print("  3. Generate Voronoi diagram")
    print("  4. Detect anomalies")
    print("  5. Create visualizations")
    print("  6. Generate interactive map (markers)")
    print("  7. Generate Voronoi overlay map (colored regions) ⭐ NEW")
    print("  8. Generate full report")
    print("  9. Run complete analysis (all steps)")
    print(" 10. Export results to CSV")
    print("  0. Exit")
    print("=" * 70)


def get_user_choice():
    """Get and validate user choice."""
    while True:
        try:
            choice = input("\nEnter your choice (0-10): ").strip()
            if choice in [str(i) for i in range(11)]:
                return int(choice)
            else:
                print("❌ Invalid choice. Please enter a number between 0 and 10.")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            return 0


def wait_for_user():
    """Wait for user to press Enter."""
    input("\nPress Enter to continue...")


def main():
    """Main execution function - Interactive Mode."""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    AIRBNB VORONOI ANOMALY DETECTOR".center(68) + "█")
    print("█" + "       Interactive Mode".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Create output directories
    create_directories()
    
    # Use the provided CSV file
    data_file = 'listings (1).csv'
    
    # Initialize variables to store analysis results
    loader = None
    df = None
    voronoi = None
    vor = None
    metrics = None
    detector = None
    viz = None
    anomaly_scores = None
    
    # Main interactive loop
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == 0:
            print("\n👋 Thank you for using the Airbnb Voronoi Anomaly Detector!")
            break
        
        # ====================================================================
        # OPTION 1: Load and analyze data
        # ====================================================================
        elif choice == 1:
            print_section("LOADING AND CLEANING DATA")
            
            if not os.path.exists(data_file):
                print(f"❌ Error: Data file '{data_file}' not found!")
                print("Please ensure the CSV file is in the current directory.")
                wait_for_user()
                continue
            
            loader = AirbnbDataLoader(data_file)
            df = loader.load_data()
            df = loader.clean_data()
            
            print("✅ Data loaded and cleaned successfully!")
            wait_for_user()
        
        # ====================================================================
        # OPTION 2: View dataset summary
        # ====================================================================
        elif choice == 2:
            print_section("DATASET SUMMARY")
            
            if loader is None or df is None:
                print("❌ Please load data first (Option 1)!")
                wait_for_user()
                continue
            
            summary = loader.get_summary()
            print(f"\n📊 Dataset Summary:")
            print(f"   • Total Listings: {summary['total_listings']}")
            print(f"   • Price Range: ${summary['price_range'][0]:.2f} - ${summary['price_range'][1]:.2f}")
            print(f"   • Average Price: ${summary['avg_price']:.2f}")
            print(f"   • Median Price: ${summary['median_price']:.2f}")
            print(f"   • Unique Neighborhoods: {summary['neighborhoods']}")
            print(f"   • Latitude Range: {summary['lat_range'][0]:.6f} to {summary['lat_range'][1]:.6f}")
            print(f"   • Longitude Range: {summary['lon_range'][0]:.6f} to {summary['lon_range'][1]:.6f}")
            wait_for_user()
        
        # ====================================================================
        # OPTION 3: Generate Voronoi diagram
        # ====================================================================
        elif choice == 3:
            print_section("GENERATING VORONOI DIAGRAM")
            
            if loader is None or df is None:
                print("❌ Please load data first (Option 1)!")
                wait_for_user()
                continue
            
            points = loader.get_coordinates()
            print(f"Processing {len(points)} coordinate points...")
            
            voronoi = VoronoiAnalyzer(points)
            vor = voronoi.generate_voronoi()
            
            bbox = loader.get_bounding_box(padding=0.02)
            
            print("Calculating spatial metrics...")
            cell_areas = voronoi.calculate_cell_areas(bound_box=bbox)
            avg_nn_dist, nn_matrix = voronoi.calculate_nearest_neighbors(k=5)
            density = voronoi.calculate_density(radius=0.01)
            
            metrics = {
                'cell_area': cell_areas,
                'nearest_neighbor': avg_nn_dist,
                'density': density,
                'price': df['price'].values
            }
            
            print("✅ Voronoi diagram and metrics calculated successfully!")
            wait_for_user()
        
        # ====================================================================
        # OPTION 4: Detect anomalies
        # ====================================================================
        elif choice == 4:
            print_section("DETECTING ANOMALIES")
            
            if metrics is None:
                print("❌ Please generate Voronoi diagram first (Option 3)!")
                wait_for_user()
                continue
            
            detector = AnomalyDetector(df, metrics)
            
            print("\nCalculating composite anomaly scores...")
            anomaly_scores = detector.composite_score(weights={
                'cell_area': 0.4,
                'nearest_neighbor': 0.3,
                'density': 0.2,
                'price': 0.1
            })
            
            print("Classifying anomalies...")
            anomaly_categories = detector.classify_anomalies(threshold=0.7)
            
            df['anomaly_score'] = anomaly_scores
            df['anomaly_category'] = anomaly_categories
            df['cell_area'] = metrics['cell_area']
            df['nearest_neighbor_dist'] = metrics['nearest_neighbor']
            df['local_density'] = metrics['density']
            
            num_anomalies = (anomaly_categories != 'Normal').sum()
            print(f"\n✅ Detected {num_anomalies} anomalies!")
            print(f"\nAnomaly breakdown:")
            for cat, count in pd.Series(anomaly_categories).value_counts().items():
                print(f"   • {cat}: {count}")
            
            wait_for_user()
        
        # ====================================================================
        # OPTION 5: Create visualizations
        # ====================================================================
        elif choice == 5:
            print_section("CREATING VISUALIZATIONS")
            
            if vor is None or 'anomaly_score' not in df.columns:
                print("❌ Please generate Voronoi diagram and detect anomalies first!")
                print("   Run options 3 and 4 before creating visualizations.")
                wait_for_user()
                continue
            
            viz = VoronoiVisualizer(vor, df, metrics)
            
            print("\n📊 Generating plots...")
            print("   • Basic Voronoi diagram...")
            viz.plot_basic_voronoi(save_path='output/figures/01_voronoi_basic.png')
            
            print("   • Voronoi colored by cell area...")
            viz.plot_colored_voronoi(color_metric='cell_area',
                                    save_path='output/figures/02_voronoi_cell_area.png')
            
            print("   • Voronoi colored by price...")
            viz.plot_colored_voronoi(color_metric='price', cmap='viridis',
                                    save_path='output/figures/03_voronoi_price.png')
            
            print("   • Anomaly detection map...")
            viz.plot_anomalies(df['anomaly_category'].values,
                             save_path='output/figures/04_anomalies_highlighted.png')
            
            print("   • Metric distributions...")
            viz.plot_metric_distributions(save_path='output/figures/05_metric_distributions.png')
            
            print("   • Price vs isolation analysis...")
            viz.plot_price_vs_isolation(df['anomaly_category'].values,
                                       save_path='output/figures/06_price_vs_isolation.png')
            
            print("\n✅ All visualizations created successfully!")
            print("📁 Check output/figures/ folder for all images")
            wait_for_user()
        
        # ====================================================================
        # OPTION 6: Generate interactive map
        # ====================================================================
        elif choice == 6:
            print_section("GENERATING INTERACTIVE MAP")
            
            if vor is None or 'anomaly_score' not in df.columns:
                print("❌ Please generate Voronoi diagram and detect anomalies first!")
                wait_for_user()
                continue
            
            if viz is None:
                viz = VoronoiVisualizer(vor, df, metrics)
            
            print("Creating interactive HTML map...")
            viz.create_interactive_map_html(
                df['anomaly_score'].values,
                df['anomaly_category'].values,
                save_path='output/maps/interactive_anomaly_map.html'
            )
            
            map_path = os.path.abspath('output/maps/interactive_anomaly_map.html')
            print(f"\n✅ Interactive map created!")
            print(f"🌐 Open in browser: file://{map_path}")
            wait_for_user()
        
        # ====================================================================
        # OPTION 7: Generate Voronoi overlay map
        # ====================================================================
        elif choice == 7:
            print_section("GENERATING VORONOI OVERLAY MAP")
            
            if vor is None or 'anomaly_score' not in df.columns:
                print("❌ Please generate Voronoi diagram and detect anomalies first!")
                wait_for_user()
                continue
            
            if viz is None:
                viz = VoronoiVisualizer(vor, df, metrics)
            
            print("\nChoose coloring method:")
            print("  1. By anomaly category (recommended)")
            print("  2. By anomaly score")
            print("  3. By price")
            
            color_choice = input("\nEnter choice (1-3): ").strip()
            
            color_by_map = {'1': 'category', '2': 'score', '3': 'price'}
            color_by = color_by_map.get(color_choice, 'category')
            
            print(f"\nCreating Voronoi overlay map colored by {color_by}...")
            print("(This may take a moment for large datasets)")
            
            viz.create_voronoi_overlay_map(
                df['anomaly_score'].values,
                df['anomaly_category'].values,
                save_path='output/maps/voronoi_overlay_map.html',
                color_by=color_by
            )
            
            map_path = os.path.abspath('output/maps/voronoi_overlay_map.html')
            print(f"\n✅ Voronoi overlay map created!")
            print(f"🌐 Open in browser: file://{map_path}")
            wait_for_user()
        
        # ====================================================================
        # OPTION 8: Generate full report
        # ====================================================================
        elif choice == 8:
            print_section("GENERATING ANALYSIS REPORT")
            
            if 'anomaly_score' not in df.columns:
                print("❌ Please complete the analysis first (Options 1-4)!")
                wait_for_user()
                continue
            
            top_anomalies = detector.get_top_anomalies(n=20)
            
            anomalies = df[df['anomaly_category'] != 'Normal']
            total_listings = len(df)
            num_anomalies = len(anomalies)
            pct_anomalies = (num_anomalies / total_listings) * 100
            
            category_counts = df['anomaly_category'].value_counts()
            
            cell_areas = metrics['cell_area']
            avg_nn_dist = metrics['nearest_neighbor']
            density = metrics['density']
            
            report = f"""
{'=' * 80}
                    AIRBNB VORONOI ANOMALY DETECTION REPORT
{'=' * 80}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {data_file}

{'=' * 80}
SUMMARY STATISTICS
{'=' * 80}

Total Listings Analyzed:     {total_listings:,}
Anomalies Detected:          {num_anomalies:,} ({pct_anomalies:.2f}%)
Normal Listings:             {total_listings - num_anomalies:,} ({100 - pct_anomalies:.2f}%)

Price Statistics:
  • Minimum:  ${df['price'].min():.2f}
  • Maximum:  ${df['price'].max():.2f}
  • Mean:     ${df['price'].mean():.2f}
  • Median:   ${df['price'].median():.2f}

Spatial Statistics:
  • Mean Cell Area:          {cell_areas[~np.isinf(cell_areas)].mean():.8f}
  • Max Cell Area:           {cell_areas[~np.isinf(cell_areas)].max():.8f}
  • Mean NN Distance:        {avg_nn_dist.mean():.6f}
  • Mean Local Density:      {density.mean():.2f} listings/radius

{'=' * 80}
ANOMALY CATEGORY BREAKDOWN
{'=' * 80}

"""
            
            for category, count in category_counts.items():
                pct = (count / total_listings) * 100
                report += f"{category:25s} {count:6,} listings ({pct:5.2f}%)\n"
            
            report += f"""
{'=' * 80}
TOP 20 MOST ANOMALOUS LISTINGS
{'=' * 80}

"""
            
            for idx, row in top_anomalies.head(20).iterrows():
                report += f"\n{'-' * 80}\n"
                report += f"Rank #{list(top_anomalies.index).index(idx) + 1}\n"
                report += f"  Name:              {row.get('name', 'N/A')[:60]}\n"
                report += f"  Neighbourhood:     {row.get('neighbourhood', 'N/A')}\n"
                report += f"  Price:             ${row.get('price', 0):.2f}\n"
                report += f"  Room Type:         {row.get('room_type', 'N/A')}\n"
                report += f"  Anomaly Score:     {row.get('anomaly_score', 0):.4f}\n"
                report += f"  Category:          {row.get('anomaly_category', 'N/A')}\n"
                area_val = row.get('cell_area', 0)
                area_str = f"{area_val:.8f}" if not np.isinf(area_val) else "Infinite (boundary)"
                report += f"  Cell Area:         {area_str}\n"
            
            report += f"""
{'=' * 80}
KEY INSIGHTS
{'=' * 80}

1. SPATIAL DISTRIBUTION
   • {(cell_areas == np.inf).sum()} listings are on the boundary with infinite cells
   • {(density < np.percentile(density, 10)).sum()} listings are in low-density areas
   • Average distance to nearest neighbor: {avg_nn_dist.mean():.6f} degrees

2. ANOMALY PATTERNS
   • Most anomalies are classified as: {category_counts.index[category_counts != category_counts['Normal']][0] if len(category_counts) > 1 else 'N/A'}
   • {(df['anomaly_score'] > 0.8).sum()} listings have very high anomaly scores (>0.8)

3. PRICE ANALYSIS
   • Anomalous listings avg price: ${anomalies['price'].mean():.2f}
   • Normal listings avg price: ${df[df['anomaly_category'] == 'Normal']['price'].mean():.2f}

4. RECOMMENDATIONS
   • Review 'Isolated' listings for potential data quality issues
   • Investigate 'Price Anomaly' listings for market opportunities
   • Monitor 'Multi-factor Anomaly' listings for suspicious patterns

{'=' * 80}
OUTPUT FILES
{'=' * 80}

Visualizations:
  • output/figures/01_voronoi_basic.png
  • output/figures/02_voronoi_cell_area.png
  • output/figures/03_voronoi_price.png
  • output/figures/04_anomalies_highlighted.png
  • output/figures/05_metric_distributions.png
  • output/figures/06_price_vs_isolation.png

Interactive Map:
  • output/maps/interactive_anomaly_map.html

Data:
  • output/reports/listings_with_anomalies.csv

{'=' * 80}
END OF REPORT
{'=' * 80}
"""
            
            report_path = 'output/reports/anomaly_report.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n📄 Report saved to: {report_path}")
            print("\n" + report)
            wait_for_user()
        
        # ====================================================================
        # OPTION 9: Run complete analysis
        # ====================================================================
        elif choice == 9:
            print_section("RUNNING COMPLETE ANALYSIS")
            print("This will run all steps automatically...\n")
            
            start_time = datetime.now()
            
            # Step 1: Load data
            print("Step 1/6: Loading data...")
            if not os.path.exists(data_file):
                print(f"❌ Error: Data file '{data_file}' not found!")
                wait_for_user()
                continue
            
            loader = AirbnbDataLoader(data_file)
            df = loader.load_data()
            df = loader.clean_data()
            print("✅ Data loaded")
            
            # Step 2: Generate Voronoi
            print("\nStep 2/6: Generating Voronoi diagram...")
            points = loader.get_coordinates()
            voronoi = VoronoiAnalyzer(points)
            vor = voronoi.generate_voronoi()
            bbox = loader.get_bounding_box(padding=0.02)
            
            cell_areas = voronoi.calculate_cell_areas(bound_box=bbox)
            avg_nn_dist, nn_matrix = voronoi.calculate_nearest_neighbors(k=5)
            density = voronoi.calculate_density(radius=0.01)
            
            metrics = {
                'cell_area': cell_areas,
                'nearest_neighbor': avg_nn_dist,
                'density': density,
                'price': df['price'].values
            }
            print("✅ Voronoi diagram generated")
            
            # Step 3: Detect anomalies
            print("\nStep 3/6: Detecting anomalies...")
            detector = AnomalyDetector(df, metrics)
            anomaly_scores = detector.composite_score(weights={
                'cell_area': 0.4,
                'nearest_neighbor': 0.3,
                'density': 0.2,
                'price': 0.1
            })
            anomaly_categories = detector.classify_anomalies(threshold=0.7)
            
            df['anomaly_score'] = anomaly_scores
            df['anomaly_category'] = anomaly_categories
            df['cell_area'] = cell_areas
            df['nearest_neighbor_dist'] = avg_nn_dist
            df['local_density'] = density
            print("✅ Anomalies detected")
            
            # Step 4: Create visualizations
            print("\nStep 4/6: Creating visualizations...")
            viz = VoronoiVisualizer(vor, df, metrics)
            
            viz.plot_basic_voronoi(save_path='output/figures/01_voronoi_basic.png')
            viz.plot_colored_voronoi(color_metric='cell_area',
                                    save_path='output/figures/02_voronoi_cell_area.png')
            viz.plot_colored_voronoi(color_metric='price', cmap='viridis',
                                    save_path='output/figures/03_voronoi_price.png')
            viz.plot_anomalies(df['anomaly_category'].values,
                             save_path='output/figures/04_anomalies_highlighted.png')
            viz.plot_metric_distributions(save_path='output/figures/05_metric_distributions.png')
            viz.plot_price_vs_isolation(df['anomaly_category'].values,
                                       save_path='output/figures/06_price_vs_isolation.png')
            print("✅ Visualizations created")
            
            # Step 5: Create interactive map
            print("\nStep 5/6: Creating interactive map...")
            viz.create_interactive_map_html(
                df['anomaly_score'].values,
                df['anomaly_category'].values,
                save_path='output/maps/interactive_anomaly_map.html'
            )
            print("✅ Interactive map created")
            
            # Step 6: Generate report
            print("\nStep 6/6: Generating report...")
            top_anomalies = detector.get_top_anomalies(n=20)
            anomalies = df[df['anomaly_category'] != 'Normal']
            
            report_path = 'output/reports/anomaly_report.txt'
            output_csv = 'output/reports/listings_with_anomalies.csv'
            df.to_csv(output_csv, index=False)
            print("✅ Report generated")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("    ✅ COMPLETE ANALYSIS FINISHED!".center(70))
            print(f"    Execution time: {duration:.2f} seconds".center(70))
            print("=" * 70)
            print("\n📁 All results saved to 'output/' folder")
            print(f"🌐 Interactive map: file://{os.path.abspath('output/maps/interactive_anomaly_map.html')}")
            wait_for_user()
        
        # ====================================================================
        # OPTION 10: Export results to CSV
        # ====================================================================
        elif choice == 10:
            print_section("EXPORTING RESULTS")
            
            if 'anomaly_score' not in df.columns:
                print("❌ Please complete the analysis first!")
                wait_for_user()
                continue
            
            output_csv = 'output/reports/listings_with_anomalies.csv'
            df.to_csv(output_csv, index=False)
            
            print(f"✅ Results exported to: {output_csv}")
            print(f"\nTotal records: {len(df)}")
            print(f"Columns exported: {', '.join(df.columns)}")
            wait_for_user()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
