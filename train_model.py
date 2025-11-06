# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import numpy as np
#
# # ===============================
# # 1️⃣ Load your dataset
# # ===============================
# df = pd.read_csv("Expanded_Destinations.csv")
#
# # ===============================
# # 2️⃣ Create 'NumberOfDays' if missing
# # ===============================
# if 'NumberOfDays' not in df.columns:
#     print("ℹ️ 'NumberOfDays' not found — generating estimated values...")
#     np.random.seed(42)
#
#     # Rough logic: based on trip type or popularity
#     if 'Type' in df.columns:
#         df['NumberOfDays'] = df['Type'].apply(lambda x: 3 if 'City' in str(x) else
#         5 if 'Hill' in str(x) else
#         7 if 'Beach' in str(x) else
#         np.random.randint(2, 8))
#     else:
#         # Default random days between 2–7 if type not available
#         df['NumberOfDays'] = np.random.randint(2, 8, len(df))
#
# print(f"✅ NumberOfDays column ready. Example values: {df['NumberOfDays'].head().tolist()}")
#
# # ===============================
# # 3️⃣ Define features and target
# # ===============================
# possible_features = ['Name', 'Name_x', 'Type', 'Preferences', 'Gender',
#                      'NumberOfAdults', 'NumberOfChildren', 'NumberOfDays']
#
# # pick whichever name exists in df
# features = [f for f in possible_features if f in df.columns]
# target = 'Popularity'
#
# if target not in df.columns:
#     raise ValueError("❌ 'Popularity' column not found in your dataset. Please check CSV.")
#
# print(f"✅ Using features: {features}")
#
# # ===============================
# # 4️⃣ Encode categorical features
# # ===============================
# label_encoders = {}
# for col in features:
#     if df[col].dtype == 'object':
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col].astype(str))
#         label_encoders[col] = le
#
# # ===============================
# # 5️⃣ Train model
# # ===============================
# X = df[features]
# y = df[target]
#
# model = RandomForestRegressor(n_estimators=150, random_state=42)
# model.fit(X, y)
#
# # ===============================
# # 6️⃣ Save model and encoders
# # ===============================
# pickle.dump(model, open('model.pkl', 'wb'))
# pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))
#
# print("🎉 Model retrained successfully with 'NumberOfDays' feature handled automatically!")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np

# ===============================
# 1️⃣ Load dataset
# ===============================
df = pd.read_csv("Expanded_Destinations.csv")

# ===============================
# 2️⃣ Handle missing or renamed columns
# ===============================
# Rename columns for consistency
if 'Name' in df.columns and 'Name_x' not in df.columns:
    df.rename(columns={'Name': 'Name_x'}, inplace=True)

# Add Preferences column if missing
if 'Preferences' not in df.columns:
    print("⚠️ 'Preferences' column missing — adding default value 'General'")
    df['Preferences'] = 'General'

# Add Gender column if missing
if 'Gender' not in df.columns:
    print("⚠️ 'Gender' column missing — adding default value 'Other'")
    df['Gender'] = 'Other'

# Add NumberOfDays column if missing
if 'NumberOfDays' not in df.columns:
    print("ℹ️ 'NumberOfDays' not found — generating estimated values...")
    np.random.seed(42)
    if 'Type' in df.columns:
        df['NumberOfDays'] = df['Type'].apply(lambda x: 3 if 'City' in str(x)
                                              else 5 if 'Hill' in str(x)
                                              else 7 if 'Beach' in str(x)
                                              else np.random.randint(2, 8))
    else:
        df['NumberOfDays'] = np.random.randint(2, 8, len(df))

# Add NumberOfAdults/Children if missing
if 'NumberOfAdults' not in df.columns:
    df['NumberOfAdults'] = np.random.randint(1, 4, len(df))
if 'NumberOfChildren' not in df.columns:
    df['NumberOfChildren'] = np.random.randint(0, 3, len(df))

# ===============================
# 3️⃣ Define features and target
# ===============================
features = ['Name_x', 'Type', 'Preferences', 'Gender',
            'NumberOfAdults', 'NumberOfChildren', 'NumberOfDays']
target = 'Popularity'

if target not in df.columns:
    raise ValueError("❌ 'Popularity' column not found in dataset!")

# ===============================
# 4️⃣ Encode categorical features
# ===============================
label_encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# ===============================
# 5️⃣ Train model
# ===============================
X = df[features]
y = df[target]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# ===============================
# 6️⃣ Save model and encoders
# ===============================
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

print("🎯 Model retrained successfully with columns:", features)
