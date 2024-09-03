# ML_learn.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_labels(df, cbi_threshold=-0.6, resolution_threshold=4, snr_threshold=None, asu_groups=None):
    """
    Create labels based on CBI values, resolution, I_obs/Sigma_obs (SNR), and ASU groups.
    
    Parameters:
    - cbi_threshold: Rows with CBI below this value are more likely to be dynamical (default=-2.0).
    - resolution_threshold: Rows with resolution above this value are more likely to be dynamical.
    - snr_threshold: Rows with SNR above this value are more likely to be dynamical.
    - asu_groups: List of ASU groups that are more likely to be dynamical (default=None).
    
    Returns:
    - DataFrame with a new 'label' column where 1 indicates dynamical effects and 0 indicates non-dynamical.
    """
    # Initial labeling based on CBI threshold
    df['label'] = (df['cbi'] < cbi_threshold).astype(int)

    # Additional labeling based on resolution if provided
    if resolution_threshold is not None:
        df['label'] = df.apply(
            lambda row: 1 if row['resolution'] > resolution_threshold else row['label'], axis=1
        )

    # Additional labeling based on SNR (I_obs / Sigma_obs) if provided
    if snr_threshold is not None:
        df['label'] = df.apply(
            lambda row: 1 if row['snr'] > snr_threshold else row['label'], axis=1
        )

    # Additional labeling based on specific ASU groups if provided
    if asu_groups is not None:
        df['label'] = df.apply(
            lambda row: 1 if (row['asu_h'], row['asu_k'], row['asu_l']) in asu_groups else row['label'], axis=1
        )

    return df


def train_and_evaluate_model(df):
    # Create labels based on CBI values
    df = create_labels(df)
    
    # Print label distribution for inspection
    print("Label distribution:", df['label'].value_counts())
    
    # Split data into features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a RandomForest classifier
    clf = RandomForestClassifier(class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Print classification report
    print(classification_report(y_test, y_pred))
    
    # Predict probabilities
    y_prob = clf.predict_proba(X_test)[:, 1]  # Get the probability for the class '1' (dynamical effects)
    
    # Combine actual labels and predicted probabilities into a DataFrame
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted_Prob': y_prob})
    
    # Optionally, set a custom threshold for classification
    threshold = 0.8  # Adjust this threshold as needed
    df_results['Predicted_Label'] = (df_results['Predicted_Prob'] >= threshold).astype(int)
    
    # Print the first few rows of the results
    print(df_results.head())
    
    # You can also print how the new predicted labels compare to the actual labels
    print(classification_report(df_results['Actual'], df_results['Predicted_Label']))
    
    return clf  # Return the trained model for further use
