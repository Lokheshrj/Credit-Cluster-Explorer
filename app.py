from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.offline as opy
import os

# Load the data
# Change this line to read an Excel file
credit_cards = pd.read_excel('credit_card_data.xlsx', engine='openpyxl')

# Drop the rows with missing values
credit_cards = credit_cards.dropna()

# Create the clusters
clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Apply KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
clusters = kmeans.fit_predict(clustering_data_scaled)
credit_cards["CREDIT_CARD_SEGMENTS"] = clusters

# Transform names of clusters for easy interpretation
credit_cards["CREDIT_CARD_SEGMENTS"] = credit_cards["CREDIT_CARD_SEGMENTS"].map({
    0: "Cluster 1",
    1: "Cluster 2",
    2: "Cluster 3",
    3: "Cluster 4"
})

# Plot the clusters using Plotly Express
# Plot the clusters using Plotly Express
fig = px.scatter_3d(credit_cards, x='BALANCE', y='PURCHASES', z='CREDIT_LIMIT',
                    color='CREDIT_CARD_SEGMENTS', symbol='CREDIT_CARD_SEGMENTS',
                    size_max=15, opacity=1, width=480, height=480,
                    title='Credit Card Segmentation')

# Save the plot as an HTML div with adjusted width and height
plot_html = opy.plot(fig, auto_open=False, output_type='div')


# Save the plot as an HTML div
plot_html = opy.plot(fig, auto_open=False, output_type='div')

# Create a Flask web application
app = Flask(__name__, static_url_path='/static',template_folder=os.path.abspath("templates"))

# Define a route to render the HTML template
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pass
    return render_template('Visualize.html', plot_html=plot_html)
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)