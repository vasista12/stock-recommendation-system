import pandas as pd
import pandas as pd

# Load the recently created CSV file
csv_file = "stocks_data.csv"  # Replace with your actual file name
try:
    df = pd.read_csv(csv_file)
    print(f"✅ Successfully loaded data from {csv_file}!")
except Exception as e:
    print(f"❌ Error reading {csv_file}: {e}")

# Display first few rows (optional)
print("\n📊 First 5 rows of stock data:")
print(df.head())

# Save the cleaned version if needed
df.to_csv("cleaned_stock_data.csv", index=False)
print("✅ Cleaned stock data saved as cleaned_stock_data.csv!")
import pandas as pd

# Load the consolidated stock CSV file
csv_file = "stocks_data.csv"  # Replace with your actual file name

try:
    # Read the CSV file
    df = pd.read_csv(csv_file)

    print(f"✅ Successfully loaded stock data from {csv_file}!\n")

    # Display first few rows
    print("\n📊 First 5 rows of stock data:")
    print(df.head())

    # Show column details
    print("\n📌 Column Information:")
    print(df.info())

except FileNotFoundError:
    print(f"❌ Error: {csv_file} not found. Please check if the file exists in the directory.")
    import pandas as pd

    # Load the consolidated stock data file
    csv_file = r"C:\Users\rvasi\PycharmProjects\pythonProject2\stocks_data.csv"  # Ensure the file is in the correct location

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        print(f"✅ Successfully loaded stock data from {csv_file}!\n")

        # Check for missing values
        print("\n📊 Missing Values in Stock Data:")
        print(df.isnull().sum())

    except FileNotFoundError:
        print(f"❌ Error: {csv_file} not found. Please check if the file exists in the directory.")
        import pandas as pd

        # Load the consolidated stock data file
        csv_file = "stocks_data.csv"  # Ensure the file is in the correct location

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)

            print(f"📊 Processing Stock Data from {csv_file}...")

            # Automatically detect the correct date column
            possible_date_cols = ["date", "Date", "timestamp", "Timestamp"]
            date_col = next((col for col in possible_date_cols if col in df.columns), None)

            if date_col:
                # Convert to datetime format
                df[date_col] = pd.to_datetime(df[date_col])

                # Set date as index
                df.set_index(date_col, inplace=True)
                print(f"✅ Date column '{date_col}' set as index!")

                # Save the modified DataFrame back to CSV
                processed_file = "Processed_" + csv_file
                df.to_csv(processed_file)
                print(f"💾 Processed data saved as '{processed_file}'")
            else:
                print(f"⚠️ No valid date column found in {csv_file}")

        except FileNotFoundError:
            print(f"❌ Error: {csv_file} not found. Please check if the file exists in the directory.")
        except Exception as e:
            print(f"❌ Error processing {csv_file}: {e}")
            import pandas as pd
            import matplotlib.pyplot as plt

            # Load the merged stock data CSV
            file_path = "stocks_data.csv"  # Update with your actual file name
            df = pd.read_csv(file_path)

            # Convert 'date' column to datetime with correct format (DD/MM/YY)
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")

            # Set 'date' as index
            df.set_index("date", inplace=True)

            # List of stock symbols
            stocks = ["AAPL", "MSFT", "TSLA"]

            # Loop through each stock and plot stock prices
            for stock in stocks:
                try:
                    # Filter data for the specific stock
                    stock_df = df[df["stock"] == stock]

                    if stock_df.empty:
                        print(f"⚠️ No data found for {stock}, skipping...")
                        continue

                    print(f"\n📊 Plotting {stock} Stock Data...")

                    # Plot closing prices
                    plt.figure(figsize=(10, 5))
                    plt.plot(stock_df.index, stock_df["4. close"], label=f"{stock} Closing Price", linestyle='-',
                             marker='o')

                    # Labeling
                    plt.xlabel("Date")
                    plt.ylabel("Stock Price")
                    plt.title(f"{stock} Stock Price Over Time")
                    plt.legend()
                    plt.grid(True)

                    # Show plot
                    plt.show()

                except Exception as e:
                    print(f"❌ Error processing {stock}: {e}")
import pandas as pd
import matplotlib.pyplot as plt

# Load the merged stock data CSV
file_path = "stocks_data.csv"  # Update with your actual file name
df = pd.read_csv(file_path)

# Convert 'date' column to datetime with correct format (DD/MM/YY)
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")

# Set 'date' as index
df.set_index("date", inplace=True)

# Get unique stock symbols dynamically
stocks = df["stock"].unique()

# Create a single plot for all stocks
plt.figure(figsize=(14, 6))

# Loop through each stock and plot stock prices
for stock in stocks:
    stock_df = df[df["stock"] == stock]

    if stock_df.empty:
        print(f"⚠️ No data found for {stock}, skipping...")
        continue

    print(f"\n📊 Plotting {stock} Stock Data...")

    # Plot closing prices with correct date handling
    plt.plot(stock_df.index, stock_df["4. close"], label=f"{stock} Closing Price", linestyle='-', marker='o')

# Labeling
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Prices Over Time")
plt.legend()
plt.grid(True)

# Show the final plot
plt.show()
import pandas as pd

# Load dataset
file_path = "stocks_data.csv"  # Ensure correct filename
df = pd.read_csv(file_path)

# Print first few rows for debugging
print("🔹 First few rows of dataset:")
print(df.head())

# Print column names
print("\n🔹 Column names:", df.columns.tolist())

# Print unique stock symbols
print("\n🔹 Unique stock symbols in dataset:", df["stock"].unique())

# Convert stock symbols to uppercase
df["stock"] = df["stock"].str.upper()

# Check if AAPL exists
if "AAPL" not in df["stock"].unique():
    raise ValueError(f"❌ 'AAPL' not found! Available symbols: {df['stock'].unique()}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import pandas as pd

# List of stock CSV files
stock_files = ["stocks_data.csv"]

# Loop through each stock file and display its data
for file in stock_files:
    try:
        # Load the stock data
        df = pd.read_csv(file)

        # Extract stock name from filename
        stock_name = file.split("_")[1]  # Correctly extracts stock symbol (AAPL, MSFT, TSLA)
        print(f"\n📄 {stock_name} Stock Data:")

        # Convert "date" column to datetime format
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
            df.set_index("date", inplace=True)  # Set "date" as index

        # Display first few rows
        print(df.head())

        # Show column details
        print("\n📌 Column Info:")
        print(df.info())

        # Check for missing values
        print("\n🔍 Missing Values:")
        print(df.isnull().sum())

    except FileNotFoundError:
        print(f"❌ Error: {file} not found. Please check if the file exists in the directory.")
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")
print(df.columns.tolist())  # Check exact column names
df.columns = df.columns.str.strip()
feature_columns = ["1. open", "2. high", "3. low", "5. volume"]
target_column = "4. close"

X = df[feature_columns].values
y = df[target_column].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data normalized!")
from sklearn.linear_model import LinearRegression

# Initialize model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("Model training complete!")
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices", color="blue")
plt.plot(y_pred, label="Predicted Prices", color="red", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")
import joblib

# Save the model
joblib.dump(model, "stock_model.pkl")
print("Model saved successfully!")
import numpy as np
import joblib

# Load the saved model
loaded_model = joblib.load("stock_model.pkl")

# Example: Use a real row from your dataset (modify as needed)
open_price = 150.0  # Replace with actual data
high_price = 155.0  # Replace with actual data
low_price = 148.0   # Replace with actual data
volume = 1000000    # Replace with actual data

# Prepare input data for prediction
new_data = np.array([[open_price, high_price, low_price, volume]])

# Make prediction
predicted_price = loaded_model.predict(new_data)
print(f"Predicted Stock Price: {predicted_price[0]}")
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))

# Save the model
import joblib
joblib.dump(model, "xgboost_stock_model.pkl")
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse}")
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset
csv_path = "stocks_data.csv"  # Ensure the CSV file is in the correct location
data = pd.read_csv(csv_path)

# Check the first few rows to understand the structure
display(data.head())

# Select relevant features and target column
features = ["1. open", "2. high", "3. low", "5. volume"]  # Adjust column names if needed
target = "4. close"  # The column we're predicting

# Drop rows with missing values (if any)
data = data.dropna()

# Prepare training and testing data
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForestRegressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save the trained model
joblib.dump(model, "stock_price_model.pkl")
print("✅ Model trained and saved successfully!")
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 🔹 Load CSV file
file_path = "/Users/lochanroyal/Downloads/stocks_data.csv"
df = pd.read_csv(file_path)

# 🔹 Print column names (debugging)
print("🔍 Columns in dataset:", df.columns.tolist())

# 🔹 Select relevant features
selected_features = ["stock", "1. open", "2. high", "3. low", "5. volume"]
df = df[selected_features].copy()

# 🔹 Drop missing values
df.dropna(inplace=True)

# 🔹 Encode the 'stock' (categorical) column
le = LabelEncoder()
df["stock"] = le.fit_transform(df["stock"])  # Convert stock symbols (AAPL, MSFT, TSLA) into numerical values

# 🔹 Define features (X) and target (y)
X = df.drop(columns=["1. open"])  # Features (exclude '1. open' if it's the target)
y = df["1. open"]  # Predicting the opening price

# 🔹 Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Mean Squared Error: {mse:.4f}")

# 🔹 Save the trained model
joblib.dump(model, "stock_price_model.pkl")
print("✅ Model trained and saved successfully!")

# 🔹 Example prediction (modify stock, high, low, and volume values)
new_data = np.array([[1, 275.17, 263.08, 3564788]])  # Example for a stock (encoded), high, low, volume
predicted_price = model.predict(new_data.reshape(1, -1))[0]
print(f"💰 Predicted Opening Price: {predicted_price:.2f}")
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 🔹 Load CSV file
file_path = "/Users/lochanroyal/Downloads/stocks_data.csv"
df = pd.read_csv(file_path)

# 🔹 Print column names for debugging
print("🔍 Columns in dataset:", df.columns.tolist())

# 🔹 Select relevant features
selected_features = ["stock", "2. high", "3. low", "5. volume"]  # Predicting '1. open'
df = df[selected_features].copy()

# 🔹 Drop missing values
df.dropna(inplace=True)

# 🔹 Encode the 'stock' (categorical) column
le = LabelEncoder()
df["stock"] = le.fit_transform(df["stock"])  # Convert stock symbols (AAPL, MSFT, TSLA) into numerical values

# 🔹 Define features (X) and target (y)
X = df  # Features (all selected features)
y = df["2. high"]  # Predicting the high price

# 🔹 Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Mean Squared Error: {mse:.4f}")

# 🔹 Save the trained model
joblib.dump(model, "stock_price_model.pkl")
print("✅ XGBoost Model trained and saved successfully!")

# 🔹 Example prediction (modify stock, high, low, and volume values)
new_data = np.array([[1, 275.17, 263.08, 3564788]])  # Example for a stock (encoded), high, low, volume
predicted_price = model.predict(new_data.reshape(1, -1))[0]
print(f"💰 Predicted Stock Price: {predicted_price:.2f}")
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# 🔹 Load CSV file
file_path = "/Users/lochanroyal/Downloads/stocks_data.csv"
df = pd.read_csv(file_path)

# 🔹 Print column names for debugging
print("🔍 Columns in dataset:", df.columns.tolist())

# 🔹 Select relevant features
selected_features = ["stock", "2. high", "3. low", "5. volume", "4. close"]  # Features + target
df = df[selected_features].copy()

# 🔹 Drop missing values
df.dropna(inplace=True)

# 🔹 Encode the 'stock' column
le = LabelEncoder()
df["stock"] = le.fit_transform(df["stock"])  # Convert stock symbols to numerical values

# 🔹 Define features (X) and target (y)
X = df[["stock", "2. high", "3. low", "5. volume"]]  # Features
y = df["4. close"]  # Target (closing price)

# 🔹 Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train the XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 🔹 Make predictions
y_pred = model.predict(X_test)

# 🔹 Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"✅ Mean Squared Error: {mse:.4f}")

# 🔹 Save the trained model
joblib.dump(model, "stock_price_model.pkl")
print("✅ XGBoost Model trained and saved successfully!")

# 🔹 Example prediction (modify input values)
new_data = np.array([[1, 275.17, 263.08, 3564788]])  # Example: stock (encoded), high, low, volume
predicted_price = model.predict(new_data.reshape(1, -1))[0]
print(f"💰 Predicted Stock Price: {predicted_price:.2f}")

# 🔹 Display Random Stock Recommendations
random_stocks = df.sample(10)[["stock", "4. close"]]
print("\n📌 Random Stock Recommendations:")
print(random_stocks)
import os

# Create folders
os.makedirs("stock_recommender/templates", exist_ok=True)
os.makedirs("stock_recommender/static", exist_ok=True)

# Create empty files
open("stock_recommender/app.py", "w").close()
open("stock_recommender/templates/index.html", "w").close()
open("stock_recommender/stocks_data.csv", "w").close()

print("✅ Project structure created!")
import pandas as pd

file_path = "/Users/lochanroyal/stock_recommender/stocks_data.csv"

try:
    df = pd.read_csv(file_path)
    print(df.head())  # Debugging: Check if data is loaded correctly
except FileNotFoundError:
    print("❌ Error: File not found. Check the file path.")
except pd.errors.EmptyDataError:
    print("❌ Error: The file is empty.")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load stock data
df = pd.read_csv("stocks_data.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y")

# Select features and target
df = df.sort_values(by="date")
df["target"] = df["4. close"].shift(-1)  # Predict next day's closing price

# Drop NaN values
df.dropna(inplace=True)

X = df[["4. close"]]  # Use past closing price as feature
y = df["target"]  # Next day closing price

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "stock_price_model.pkl")

print("✅ Model trained and saved as 'stock_price_model.pkl'")
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("stocks_data.csv")
df["Stock Price"] = np.random.uniform(100, 500, len(df))  # Dummy prices

X = df[["5. volume"]]  # Use an existing column
y = df["4. close"]  # Predict closing price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X_train, y_train)

joblib.dump(model, "stock_price_model.pkl")
print("✅ Model saved!")
import pandas as pd

df = pd.read_csv("stocks_data.csv")
print(df.columns)
import pandas as pd

df = pd.read_csv("stocks_data.csv")
print(df.columns)
