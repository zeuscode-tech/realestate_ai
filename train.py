import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

df = pd.read_csv("Real estate.csv")
df.columns = [c.strip() for c in df.columns]
df = df.rename(columns={'Y house price of unit area': 'price', 'X3 distance to the nearest MRT station': 'mrt'})

df = df[df["price"] < df["price"].quantile(0.99)]

df['mrt_log'] = np.log1p(df['mrt'])

df['area_mean'] = df.groupby([df['X5 latitude'].round(3), df['X6 longitude'].round(3)])['price'].transform('mean')


X = df[['X2 house age', 'mrt_log', 'X4 number of convenience stores', 'area_mean']]
y = df['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)


area_map = df.groupby([df['X5 latitude'].round(3), df['X6 longitude'].round(3)])['price'].mean().to_dict()

with open("re_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "area_map": area_map,
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "r2": float(r2)
    }, f)

print("--- СИСТЕМА: Файл re_model.pkl успешно создан! ---")

print(f"\n" + "🚀" * 20)
print(f"ИТОГОВЫЙ R2: {r2*100:.2f}%")
print(f"🚀" * 20 + "\n")


plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3)
plt.title(f"Точность = {r2*100:.2f}%")
plt.xlabel("Реальность")
plt.ylabel("Предсказание")
plt.savefig("templates/scatter.png", dpi=100, bbox_inches='tight', facecolor='#1e293b')
plt.show()
