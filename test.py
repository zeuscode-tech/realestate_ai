import numpy as np
import pickle
import os
import sys

print("--- ЗАПУСК ТЕСТА ---")


file_name = "re_model.pkl"
if not os.path.exists(file_name):
    print(f"❌ ОШИБКА: Файл '{file_name}' не найден!")
    print(f"Текущая папка: {os.getcwd()}")
    input("Нажми Enter для выхода...")
    sys.exit()


print(f"⌛ Загрузка модели из {file_name}...")
try:
    with open(file_name, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    area_map = data["area_map"]
    print("✅ Модель успешно загружена!")
except Exception as e:
    print(f"❌ Ошибка при чтении: {e}")
    input("Нажми Enter...")
    sys.exit()


print("\n--- ВВЕДИТЕ ДАННЫЕ КВАРТИРЫ ---")
try:
    age = float(input("1. Возраст дома (лет): "))
    mrt = float(input("2. Метры до метро: "))
    stores = int(input("3. Магазинов рядом: "))
    lat = float(input("4. Широта (например, 24.97): "))
    lon = float(input("5. Долгота (например, 121.54): "))

    mrt_log = np.log1p(mrt)
    coords = (round(lat, 3), round(lon, 3))
    area_mean = area_map.get(coords, 38.0) 

    user_input = np.array([[age, mrt_log, stores, area_mean]])
    price_pred = model.predict(user_input)[0]

    price_twd_per_ping = price_pred * 10000  
    price_twd_per_m2 = price_twd_per_ping / 3.3  
    usd_rate = 31  
    price_usd_per_ping = price_twd_per_ping / usd_rate  
    price_usd_per_m2 = price_twd_per_m2 / usd_rate  

    print(f"\n💰 ПРОГНОЗ ЦЕНЫ:")
    print(f"  ├─ {price_twd_per_ping:>10,.0f} TWD за пинг (坪)")
    print(f"  ├─ {price_twd_per_m2:>10,.0f} TWD за м²")
    print(f"  ├─ {price_usd_per_ping:>10,.0f} USD за пинг")
    print(f"  └─ {price_usd_per_m2:>10,.0f} USD за м²")

except ValueError:
    print("❌ Ошибка: нужно вводить только цифры!")
