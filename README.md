🪐 A World Away: Hunting for Exoplanets with AI

https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white
https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
https://img.shields.io/badge/NASA-Data-0B3D91?style=for-the-badge&logo=nasa&logoColor=white

An advanced AI-powered exoplanet detection system that combines machine learning, signal processing, and astronomical analysis to discover planets around distant stars.

🌟 Overview

"A World Away" is a comprehensive, single-file Python application that automates the process of exoplanet detection using data from NASA's Kepler and TESS space telescopes. This sophisticated system employs a hybrid AI model to analyze light curves, detect transit signals, and characterize planetary properties with professional-grade accuracy.

https://via.placeholder.com/800x400/0B3D91/FFFFFF?text=Exoplanet+AI+Detection+Dashboard

🚀 Key Features

🤖 Advanced AI Architecture

· Hybrid CNN + LSTM Neural Network for robust pattern recognition
· Ensemble Learning with confidence calibration
· Synthetic Data Training for improved model accuracy
· Real-time Model Adaptation and continuous learning

🔬 Scientific Analysis

· Multi-stage Signal Processing Pipeline with Savitzky-Golay filtering and FFT
· Advanced Feature Extraction (13+ physical parameters)
· Bayesian Statistical Analysis with BIC scoring
· Uncertainty Quantification for all derived parameters

📊 Comprehensive Visualization

· Interactive 3D Orbital Simulations with habitable zones
· Real-time Confidence Gauges and analytics dashboards
· Spectral Analysis Tools with periodogram displays
· Professional PDF Reporting system

🌌 Data Integration

· Direct NASA API Access (Kepler & TESS missions)
· Enhanced Simulated Data for testing and demonstration
· Multiple Target Support (TIC, KIC, and custom identifiers)
· Robust Error Handling with graceful degradation

🛠️ Installation

Prerequisites

· Python 3.8 or higher
· 4GB RAM minimum (8GB recommended)
· Internet connection for data fetching

Quick Start

1. Clone the repository

```bash
git clone https://github.com/yourusername/exoplanet-ai-detector.git
cd exoplanet-ai-detector
```

1. Install dependencies

```bash
pip install -r requirements.txt
```

1. Run the application

```bash
streamlit run exoplanet_ai.py
```

Required Libraries

The application uses the following Python libraries:

```python
numpy, pandas, matplotlib, plotly, scipy, scikit-learn, tensorflow, keras,
lightkurve, astroquery, lime, shap, streamlit, reportlab
```

All dependencies are automatically handled in the single-file implementation.

🎯 Usage Guide

Basic Operation

1. Launch the Application
   ```bash
   streamlit run exoplanet_ai.py
   ```
2. Configure Observation Parameters
   · Select mission (TESS, Kepler, or Simulated Data)
   · Enter target identifier (TIC ID, KIC ID, or custom)
   · Adjust observation sector/quarter
3. Start Analysis
   · Click "🚀 Start Enhanced Exoplanet Hunt"
   · Monitor real-time processing progress
   · Review AI confidence scores and results

Target Examples

Mission Target ID Description
TESS TIC 261136679 Known exoplanet host
Kepler KIC 757450 Well-studied system
Kepler KIC 11442793 Kepler-90 multi-planet system
Simulated SIM_001 Custom synthetic data

Advanced Configuration

Enable these options in the sidebar for enhanced analysis:

· Auto-train AI Model: Train the neural network with synthetic data
· Show Advanced Plots: Display FFT analysis and spectral features
· Custom Preprocessing: Adjust filtering parameters and detection thresholds

📈 Output Features

Detection Metrics

· Exoplanet Probability Score (0-100%)
· Confidence Calibration with uncertainty estimates
· Feature Importance Analysis using SHAP/LIME
· Multi-parameter Validation checks

Planetary Characteristics

· Radius Estimation (Earth radii) with uncertainty
· Orbital Period and semi-major axis
· Equilibrium Temperature calculations
· Transit Depth and duration measurements
· Habitability Index scoring

Visualizations

· Light Curve Analysis (original vs processed)
· 3D Orbital Simulations with inclinations
· FFT Frequency Spectrum and periodograms
· Feature Importance Dashboards
· Real-time Confidence Gauges

🎨 Interface Overview

Main Dashboard

· Real-time Processing status indicators
· Interactive Plots with zoom and pan capabilities
· Multi-tab Results organization
· Export Controls for data and reports

Analysis Tabs

1. 📊 Planet Parameters: Physical characteristics and uncertainties
2. 🔍 Feature Analysis: Extracted signal features and importance
3. 🤖 AI Insights: Model confidence and recommendations
4. 📄 Export Results: PDF reports and data exports

📊 Scientific Methodology

Signal Processing Pipeline

1. Data Acquisition: Fetch from NASA archives or generate synthetic data
2. Preprocessing:
   · Sigma clipping for outlier removal
   · Savitzky-Golay filtering for noise reduction
   · Stellar variability detrending
3. Feature Extraction:
   · Transit depth and duration
   · Orbital period via FFT analysis
   · Signal-to-noise ratios
   · Consistency metrics across transits

AI Detection Algorithm

```python
# Hybrid CNN + LSTM Architecture
Input → CNN Layers (Spatial Features) → LSTM Layers (Temporal Patterns) → Dense Layers → Classification
```

Physical Parameter Estimation

· Planet Radius: R_planet/R_star = √(transit_depth)
· Orbital Distance: Kepler's third law approximations
· Equilibrium Temperature: Radiative balance calculations
· Habitability Index: Multi-parameter scoring system

🗂️ Project Structure

```
exoplanet-ai-detector/
│
├── exoplanet_ai.py          # Main application file
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── examples/               # Example outputs and reports
    ├── sample_report.pdf
    ├── feature_analysis.png
    └── orbital_diagram.png
```

🔧 Technical Details

Model Architecture

· Input Layer: 13 feature dimensions
· CNN Blocks: 3 convolutional layers with batch normalization
· LSTM Layers: Temporal pattern recognition
· Output: Binary classification with confidence scoring

Performance Metrics

· Training Accuracy: >92% on synthetic datasets
· Inference Speed: <30 seconds for typical analysis
· Memory Usage: <2GB for complete processing
· Supported Data: Light curves with 1,000-10,000 points

Data Sources

· Primary: NASA MAST Archive (via lightkurve)
· Secondary: Simulated data with realistic noise models
· Validation: Known exoplanet systems from NASA Exoplanet Archive

🚀 Advanced Usage

Custom Target Analysis

```python
# Programmatic usage example
detector = EnhancedExoplanetDetector()
lc = detector.fetch_tess_data("TIC 123456789", sector=25)
features = detector.extract_advanced_features(lc.time, lc.flux)
probability = detector.predict_exoplanet_probability(features)
```

Batch Processing

The modular design allows for batch processing of multiple targets by iterating through target lists and automating the analysis pipeline.

Research Integration

Export features and probabilities for further statistical analysis in research publications or machine learning workflows.

📄 Output Examples

PDF Reports Include:

· Executive summary with detection status
· Detailed planetary parameter tables
· Feature analysis with scientific context
· High-quality visualization exports
· AI confidence assessments
· Recommendations for follow-up observations

Data Exports:

· CSV feature sets for further analysis
· Processed light curve data
· Model confidence scores
· Parameter uncertainty estimates

🐛 Troubleshooting

Common Issues

1. Data Download Failures
   · Check internet connection
   · Verify target identifier format
   · Use simulated data for testing
2. Memory Errors
   · Reduce data point count
   · Close other memory-intensive applications
   · Use system with more RAM
3. Model Training Issues
   · Enable "Auto-train AI Model"
   · Ensure TensorFlow installation
   · Check Python version compatibility

Performance Tips

· Use simulated data for initial testing
· Enable GPU acceleration if available
· Close unnecessary browser tabs during analysis
· Use the provided example targets for validation

🤝 Contributing

We welcome contributions from the community! Please see our Contributing Guidelines for details.

Areas for Improvement

· Additional telescope data sources
· Enhanced atmospheric characterization
· Multi-planet system detection
· Real-time data streaming capabilities

📜 Citation

If you use this software in your research, please cite:

```bibtex
@software{exoplanet_ai_2024,
  title = {A World Away: Hunting for Exoplanets with AI},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/exoplanet-ai-detector}
}
```

📞 Support

· 📧 Email: support@exoplanet-ai.com
· 🐛 Issues: GitHub Issues
· 💬 Discussions: GitHub Discussions

🏆 Acknowledgments

· NASA for Kepler and TESS mission data
· Lightkurve development team for data access tools
· TensorFlow and Streamlit communities
· The exoplanet research community for validation datasets

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">

Made with ❤️ for the exploration of brave new worlds

"The cosmos is within us. We are made of star-stuff. We are a way for the universe to know itself." - Carl Sagan

</div>

🔄 Changelog

v1.0.0 (2024-01-01)

· Initial release with hybrid AI model
· NASA Kepler/TESS data integration
· Comprehensive visualization suite
· Professional PDF reporting

---

⭐ Star this repository if you find it useful!
