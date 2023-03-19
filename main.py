import os
import mysql.connector
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from pyngrok import ngrok
import plotly.offline as pyo
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import itertools

# Parameters
size_threshold = 10
penalty_factor = 5
distance_threshold = 50  # Change this value to control the number of clusters
magnitude_threshold = 8  # Magnitude threshold for applying the penalty
magnitude_penalty_factor = 5  # Penalty factor to be added to the distance when the magnitude threshold is exceeded

# Connect to your MySQL database
cnx = mysql.connector.connect(user=os.environ['mysql_username'],
                              password=os.environ['mysql_password'],
                              host=os.environ['mysql_host'],
                              database=os.environ['mysql_db'])

# Read the star data from the MySQL table
query = "SELECT x, y, z, iauname, altname, bf, gl, absmag FROM hyg WHERE dist < 32.6"
df = pd.read_sql(query, cnx)
cnx.close()

# Calculate the distance matrix using Euclidean distances
dist_matrix = pdist(df[['x', 'y', 'z']].values, metric='euclidean')
dist_matrix = squareform(dist_matrix)

# Apply the magnitude penalty to the distance matrix
for i in range(df.shape[0]):
  for j in range(i + 1, df.shape[0]):
    if df.iloc[i]['absmag'] > magnitude_threshold or df.iloc[j][
        'absmag'] > magnitude_threshold:
      dist_matrix[i, j] += magnitude_penalty_factor
      dist_matrix[j, i] = dist_matrix[i, j]

# Perform hierarchical clustering using the complete-linkage method
#clusters = linkage(dist_matrix, method='complete')
condensed_dist_matrix = squareform(dist_matrix, checks=False)
clusters = linkage(condensed_dist_matrix, method='complete')

# Determine the cluster labels using the distance_threshold
labels = fcluster(clusters, distance_threshold, criterion='distance')


def normalize_marker_size(sizes, min_size=2, max_size=9):
  sizes_no_nan = sizes[~np.isnan(sizes)]
  min_val, max_val = min(sizes_no_nan), max(sizes_no_nan)
  normalized_sizes = min_size + (max_size - min_size) * (1 -
                                                         (sizes - min_val) /
                                                         (max_val - min_val))
  normalized_sizes[np.isnan(sizes)] = min_size
  return normalized_sizes


# Helper function to get the combined name from the available columns
def get_combined_name(row):
  for col in ['iauname', 'altname', 'bf', 'gl']:
    if not pd.isna(row[col]):
      return row[col]
  return ''


# Create an empty 3D scatter plot
fig = go.Figure()

# Assign a unique color to each cluster
colors = px.colors.qualitative.Plotly

json_clusters = []

for label in set(labels):
  cluster_indices = np.where(labels == label)[0]
  cluster_data = df.iloc[cluster_indices]
  # Add the combined name column to a new DataFrame
  cluster_data = cluster_data.assign(
    name=cluster_data.apply(get_combined_name, axis=1))

  # Determine the name of the cluster based on the brightest star's combined name
  brightest_star = cluster_data.loc[cluster_data['absmag'].idxmin()]
  cluster_name = f"{brightest_star['name']} Cluster"

  # Normalize the marker size based on the star's brightness
  marker_sizes = normalize_marker_size(8 - cluster_data['absmag'])
  marker_sizes = np.nan_to_num(marker_sizes, nan=2)

  # Add points from the current cluster to the plot
  scatter = go.Scatter3d(x=cluster_data['x'],
                         y=cluster_data['y'],
                         z=cluster_data['z'],
                         mode='markers+text',
                         marker=dict(size=marker_sizes,
                                     color=colors[label % len(colors)]),
                         name=cluster_name,
                         textposition="top center")

  # Add text labels for bright stars
  scatter.text = cluster_data.apply(lambda row: row['name']
                                    if row['absmag'] < 8.0 else '',
                                    axis=1)

  fig.add_trace(scatter)

  # Draw lines between stars in the same cluster closer than 8 light years
  line_x = []
  line_y = []
  line_z = []
  for star1, star2 in itertools.combinations(cluster_data.iterrows(), 2):
    distance = np.sqrt((star1[1]['x'] - star2[1]['x'])**2 +
                       (star1[1]['y'] - star2[1]['y'])**2 +
                       (star1[1]['z'] - star2[1]['z'])**2)
    if distance < 8:
      line_x += [star1[1]['x'], star2[1]['x'], None]
      line_y += [star1[1]['y'], star2[1]['y'], None]
      line_z += [star1[1]['z'], star2[1]['z'], None]

  # Add the lines as a separate trace
  lines = go.Scatter3d(x=line_x,
                       y=line_y,
                       z=line_z,
                       mode='lines',
                       line=dict(color=colors[label % len(colors)], width=1),
                       showlegend=False)
  fig.add_trace(lines)

  # Add the cluster information to the JSON output
  cluster_info = {
    'star_count':
    len(cluster_data),
    'stars':
    cluster_data[['name', 'x', 'y', 'z', 'absmag']].to_dict(orient='records')
  }
  json_clusters.append(cluster_info)

fig.update_layout(scene=dict(xaxis_title='X (light years)',
                             yaxis_title='Y (light years)',
                             zaxis_title='Z (light years)'))

# Save the JSON output to a file
import json
with open('clusters.json', 'w') as json_file:
  json.dump(json_clusters, json_file, indent=2)

# Save the plot to an HTML file
plot_file = 'plot.html'
pyo.plot(fig, filename=plot_file, auto_open=False)

# Serve the HTML file using a temporary public URL
public_url = ngrok.connect(8080)
print(f"Plotly plot is available at: {public_url}/plot.html")

# Serve the file using Python's built-in HTTP server
subprocess.Popen(["python", "-m", "http.server", "8080", "--bind", "0.0.0.0"])
