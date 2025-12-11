import pandas as pd
import matplotlib.pyplot as plt

csv_path = "LTA1_orientations_scored.csv"   # <- change to your actual path
df = pd.read_csv(csv_path)

for col in ["image", "problematic_score"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["image", "problematic_score"]).sort_values("image")

window = 25  # tweak
df["ps_roll"] = df["problematic_score"].rolling(window, center=True, min_periods=1).mean()

plt.figure()
plt.scatter(df["image"], df["problematic_score"], s=8)
plt.plot(df["image"], df["ps_roll"], linewidth=2)
plt.axvline(130, linestyle="--")
plt.xlabel("Frame (image)")
plt.ylabel("Problematic score")
plt.title(f"Problematic score vs frame (rolling mean window={window})")
plt.tight_layout()
plt.show()
