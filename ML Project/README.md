# Autism Prediction Dashboard

An interactive dashboard for visualizing and analyzing autism screening data.

## Features

- Interactive visualizations of autism screening data
- Filter data by gender and ethnicity
- View key metrics and statistics
- Analyze score patterns and correlations
- Responsive and user-friendly interface

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Make sure your dataset file (`Autism-Prediction-using-Machine-Learning---DataSet.csv`) is in the same directory as `app.py`

## Running the Dashboard

To run the dashboard, execute:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## Usage

- Use the sidebar filters to select specific genders and ethnicities
- Interact with the charts by hovering over data points
- Click on legend items to show/hide specific data series
- Use the zoom and pan tools in the charts for detailed exploration

## Data Visualization Components

1. **Key Metrics**
   - Total number of cases
   - Autism prevalence percentage
   - Average age of participants

2. **Distribution Charts**
   - Age distribution
   - Gender distribution
   - Ethnicity distribution
   - Autism prevalence by ethnicity

3. **Score Analysis**
   - Average scores by question
   - Correlation heatmap of scores

## Contributing

Feel free to submit issues and enhancement requests! 