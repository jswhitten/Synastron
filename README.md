# Synastron

Synastron is a Python script to generate 3D clusters of nearby stars called synastrons, based on their spatial proximity. The script uses hierarchical clustering to create well-defined groups of stars, and outputs the results as json and as an interactive plot.

## Features

- Reads star data from a MySQL database
- Uses agglomerative hierarchical clustering to group stars based on their spatial proximity
- Applies penalties to prevent clusters from becoming too large
- Generates an interactive 3D plot of the clusters using Plotly
- Visualizes clusters with unique colors, different marker sizes based on brightness, and lines connecting stars within a cluster that are closer than a specified distance
- Serves the 3D plot using a temporary public URL for easy sharing

## Usage

1. Clone the repository to your local machine or import it to Replit.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Set the following environment variables for your MySQL database connection:
   - `mysql_username`
   - `mysql_password`
   - `mysql_host`
   - `mysql_db`
4. Run the script using `python main.py`.
5. Open the generated temporary public URL to view the 3D plot of the clusters.

## Dependencies

- mysql-connector-python
- pandas
- numpy
- scipy
- pyngrok
- plotly
- matplotlib
- itertools
