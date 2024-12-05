from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# File paths
DATA_FILE = "model/car_data.pkl"
MODEL_FILE = "model/model.pkl"

# Ensure the model directory exists
os.makedirs("model", exist_ok=True)

# Initialize data
if os.path.exists(DATA_FILE):
    car_data = joblib.load(DATA_FILE)
else:
    car_data = pd.DataFrame(columns=["Year", "Hand", "Price"])
    joblib.dump(car_data, DATA_FILE)

# Initialize model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = LinearRegression()
    joblib.dump(model, MODEL_FILE)


@app.route("/", methods=["GET", "POST"])
def index():
    global car_data, model

    if request.method == "POST":
        try:
            # Collect data from the form
            year = int(request.form["year"])
            hand = int(request.form["hand"])
            price = float(request.form["price"])

            # Add new data
            new_data = pd.DataFrame({"Year": [year], "Hand": [hand], "Price": [price]})
            car_data = pd.concat([car_data, new_data], ignore_index=True)
            joblib.dump(car_data, DATA_FILE)

            # Train the model
            if len(car_data) > 1:
                X = car_data[["Year", "Hand"]]
                y = car_data["Price"]
                model.fit(X, y)
                joblib.dump(model, MODEL_FILE)

        except Exception as e:
            print(f"Error during data processing: {e}")

    # Create graph - image save to static  and display in the site
    plt.figure(figsize=(10, 6))
    try:
        plt.scatter(car_data["Year"], car_data["Price"], label="Data Points", color="blue")
        if len(car_data) > 1:
            unique_years = sorted(car_data["Year"].unique())
            avg_hand = car_data["Hand"].mean()
            predictions = model.predict([[y, avg_hand] for y in unique_years])
            plt.plot(unique_years, predictions, label="Regression Line", color="red")

        plt.xlabel("Year")
        plt.ylabel("Price")
        plt.title("Car Price Prediction")
        plt.legend()
        plt.grid(True)

        # Save plot
        plot_path = "static/plot.png"
        plt.savefig(plot_path)
    except Exception as e:
        print(f"Error during plotting: {e}")
    finally:
        plt.close()

    return render_template("index.html", plot_url="static/plot.png")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    global model

    if request.method == "POST":
        try:
            year = int(request.form["year"])
            hand = int(request.form["hand"])

            # Predict price
            prediction = model.predict([[year, hand]])[0]
            return render_template("predict.html", year=year, hand=hand, price=round(prediction, 2))
        except Exception as e:
            print(f"Error during prediction: {e}")
            return redirect(url_for("index"))

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
