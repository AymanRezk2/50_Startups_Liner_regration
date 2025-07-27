from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# load the pre-trained model
model = joblib.load("model_pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # input from the form
            RnD_Spend = float(request.form["RnD_Spend"])
            Administration = float(request.form["Administration"])
            Marketing_Spend = float(request.form["Marketing_Spend"])
            State = request.form["State"]

            # convert categorical variable 'State' to numerical
            input_df = pd.DataFrame(
                [[RnD_Spend, Administration, Marketing_Spend, State]],
                columns=["R&D Spend", "Administration", "Marketing Spend", "State"]
            )

            # predict using the model
            prediction = model.predict(input_df)
            output = round(float(prediction[0]), 2)

            return render_template("index.html",
                                prediction_text=f"Predicted Profit: ${output}")

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template("index.html",
                                prediction_text="Error in input. Please try again.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
