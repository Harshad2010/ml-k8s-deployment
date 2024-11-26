import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Train the model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)
