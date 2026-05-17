import pandas as pd
from joblib import load

loaded_model=load("model.joblib")

test_circuits={
  "3-to-8 Decoder" : [10,3,1],
  "2-bit Comparator" : [8,2,2],
  "4-bit Priority Encoder" : [12,3,1],
  "2-bit ALU" : [16,3,2],
  "2-bit Up/Down Counter" : [10,2,2]
}

test_df= pd.DataFrame.from_dict(test_circuits, orient="index", columns=["Gates", "Fan-In", "Fan-Out"])

predicted_depths=loaded_model.predict(test_df)

print("\nPredicted Combinational Depth for NEew RTL Designs:")
for circuit, depth in zip(test_circuits.keys(), predicted_depths):
  print(f"{circuit}: {depth:.2f}")
