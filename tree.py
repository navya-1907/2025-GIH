from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import matplotlib.pyplot as plt
import pydotplus

# Load dataset
df = pd.read_csv("dataset.csv")

# Features and target
X = df[['Gates', 'Fan-In', 'Fan-Out']]
y = df['Depth']

# Train model
model = DecisionTreeRegressor()
model.fit(X, y)

# Export tree data
dot_data = tree.export_graphviz(
    model,
    out_file=None,
    feature_names=X.columns,
    filled=True,
    rounded=True
)

# Create graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Save png
graph.write_png("tree.png")

# Read image
image = plt.imread("tree.png")

# Show image
plt.figure(figsize=(14,10))
plt.imshow(image)
plt.axis('off')
plt.show()
