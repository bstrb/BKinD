#main.py


import random
import shap

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from parse_xds_ascii import parse_xds_ascii
from feature_engineering import feature_engineering
from ML_learn import train_and_evaluate_model, create_labels

def iterative_learning_pipeline(df, iterations=10, ask_user=True):
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")

        # Split data into training and testing sets
        X = df.drop(columns=['label'])  # Assuming 'label' is the target column
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        print(f"Iteration {iteration + 1} Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))

        if ask_user:
            # Ask user to validate some predictions
            for idx in range(len(y_test)):
                if np.random.rand() < 0.1:  # Randomly select a subset to ask user
                    actual_label = y_test.iloc[idx]
                    predicted_label = y_pred[idx]
                    confidence = max(model.predict_proba([X_test.iloc[idx]])[0])

                    response = simpledialog.askstring("User Validation",
                                                      f"Model predicted {predicted_label} with confidence {confidence:.2f}. "
                                                      f"Actual was {actual_label}. Is this correct? (y/n)")

                    if response.lower() == 'n':
                        # If user disagrees, correct the model
                        corrected_label = simpledialog.askinteger("Correct Label", "What is the correct label?")
                        y_train = y_train.append(pd.Series(corrected_label), ignore_index=True)
                        X_train = X_train.append(X_test.iloc[idx])

                        # Retrain model with updated data
                        model.fit(X_train, y_train)
                        print(f"Model retrained after user correction at iteration {iteration + 1}.")

        # Feature importance and model interpretability
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test)

    return model

class XDSAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("XDS ASCII Analyzer")

        self.file_path = ""
        self.create_widgets()

    def create_widgets(self):
        self.browse_button = tk.Button(self.root, text="Browse XDS_ASCII_CBI_ASU_RES.HKL", command=self.browse_file)
        self.browse_button.pack(pady=10)

        self.run_button = tk.Button(self.root, text="Run Analysis", command=self.run_analysis, state=tk.DISABLED)
        self.run_button.pack(pady=10)

    def browse_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("XDS ASCII Files", "*.HKL")], title="Select an XDS_ASCII_CBI_ASU_RES.HKL file")
        
        if not self.file_path:
            messagebox.showwarning("No file selected", "Please select an XDS_ASCII_CBI_ASU_RES.HKL file.")
            return

        self.run_button.config(state=tk.NORMAL)

    def run_analysis(self):
        
        def make_predictions(unlabeled_data, model, threshold=0.8):
            """
            Make predictions on unlabeled data with a confidence threshold.
            Only return predictions below the threshold for user validation.
            """
            predictions = model.predict_proba(unlabeled_data)[:, 1]  # Probability of dynamical effect
            results = []
            uncertain_indices = []

            for i, prob in enumerate(predictions):
                if prob >= threshold:
                    results.append(1)
                elif prob <= 1 - threshold:
                    results.append(0)
                else:
                    results.append(None)  # Uncertain, requires user validation
                    uncertain_indices.append(i)
            
            return results, uncertain_indices

        if not self.file_path:
            print("No file selected. Please select an XDS_ASCII_CBI_ASU_RES.HKL file.")
            return

        # Step 1: Parse the XDS_ASCII_CBI_ASU_RES file
        try:
            df = parse_xds_ascii(self.file_path)
            print("Parsing Completed: Successfully parsed the XDS_ASCII_CBI_ASU_RES.HKL file.")
        except Exception as e:
            print(f"Parsing Error: An error occurred during parsing: {e}")
            return

        # Step 2: Perform feature engineering
        try:
            df = feature_engineering(df)
            print("Feature Engineering Completed: Successfully performed feature engineering.")
        except Exception as e:
            print(f"Feature Engineering Error: An error occurred during feature engineering: {e}")
            return

        # Step 3: Create initial labels based on CBI values
        try:
            df = create_labels(df)
            print("Initial Labels Created Based on CBI values.")
        except Exception as e:
            print(f"Labeling Error: An error occurred during labeling: {e}")
            return

        # Step 4: Interactive Learning Process
        try:
            # Randomly sample 10 data points for manual labeling
            unlabeled_indices = df[df['label'].isna()].index.tolist()
            sample_size = min(10, len(unlabeled_indices))  # Ensure we don't sample more than available
            random_indices = random.sample(unlabeled_indices, sample_size)
            
            for i in random_indices:
                row = df.iloc[i]
                response = messagebox.askquestion(
                    "Label Validation",
                    f"Row {i}:\n{row}\nIs this a dynamical effect? (yes/no/maybe)"
                )
                if response == 'yes':
                    df.at[i, 'label'] = 1
                elif response == 'no':
                    df.at[i, 'label'] = 0
                else:  # 'maybe'
                    continue  # Skip labeling this data point for now

            # Initial training
            model = train_and_evaluate_model(df)

            # Limit the number of points reviewed in each iteration
            max_points_per_iteration = 10

            # Continue the iterative learning process
            while True:
                labeled_data = df.dropna(subset=['label'])
                if labeled_data.empty:
                    print("No labeled data. Please label some data before training.")
                    return
                
                model = train_and_evaluate_model(labeled_data)  # Re-train the model on updated labels

                # Predict on the remaining unlabeled data
                unlabeled_data = df[df['label'].isna()].drop('label', axis=1)
                if unlabeled_data.empty:
                    print("All data has been labeled and analyzed.")
                    break

                predicted_labels, uncertain_indices = make_predictions(unlabeled_data, model)

                # Randomly select a subset of uncertain predictions to ask the user
                if uncertain_indices:
                    selected_indices = random.sample(uncertain_indices, min(len(uncertain_indices), max_points_per_iteration))

                    # Ask the user to validate selected uncertain predictions
                    for i in selected_indices:
                        index = unlabeled_data.index[i]
                        row = df.loc[index]
                        response = messagebox.askquestion(
                            "Label Validation",
                            f"Row: {row}\nPredicted as dynamical effect (yes) or non-dynamical (no). Is this correct? (yes/no/maybe)"
                        )
                        if response == 'yes':
                            df.at[index, 'label'] = 1
                        elif response == 'no':
                            df.at[index, 'label'] = 0
                        else:  # 'maybe'
                            continue  # Skip this data point for now

                if messagebox.askyesno("Continue?", "Do you want to continue the iterative learning process?") == False:
                    break

            print("Analysis Completed: Interactive analysis completed successfully.")
        except Exception as e:
            print(f"ML Error: An error occurred during the learning process: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = XDSAnalyzerApp(root)
    root.mainloop()
