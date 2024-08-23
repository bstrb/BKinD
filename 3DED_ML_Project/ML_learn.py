# ML_learn.py

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from parse_xds_ascii import parse_xds_ascii
from feature_engineering import feature_engineering
from prepare_dataframe import prepare_dataframe

def create_labels(df):
    # Group by frames to calculate the total number of reflections and sum of intensities per frame
    frame_stats = df.groupby('zd').agg({
        'iobs': ['sum', 'count'],
    })
    frame_stats.columns = ['total_intensity', 'reflection_count']
    
    # Merge these stats back into the original DataFrame
    df = df.merge(frame_stats, left_on='zd', right_index=True)
    
    # Define a threshold for labeling
    intensity_threshold = df['total_intensity'].quantile(0.75)  # Top 25% in intensity
    reflection_count_threshold = df['reflection_count'].quantile(0.75)  # Top 25% in count

    # Create a label: 1 if both intensity and reflection count are high, otherwise 0
    df['labels'] = ((df['total_intensity'] > intensity_threshold) & 
                    (df['reflection_count'] > reflection_count_threshold)).astype(int)
    
    return df
def train_and_evaluate_model(df):
    # Create labels before training
    df = create_labels(df)
    
    # Print label distribution for inspection
    print("Label distribution:", df['labels'].value_counts())
    
    # Split data into features and target
    X = df.drop('labels', axis=1)
    y = df['labels']
    
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
    threshold = 0.95  # Default is 0.5, but you can adjust it
    df_results['Predicted_Label'] = (df_results['Predicted_Prob'] >= threshold).astype(int)
    
    # Print the first few rows of the results
    print(df_results.head())
    
    # You can also print how the new predicted labels compare to the actual labels
    print(classification_report(df_results['Actual'], df_results['Predicted_Label']))

if __name__ == "__main__":
    # Example DataFrame creation or loading
    df = prepare_dataframe(parse_xds_ascii('XDS_ASCII.HKL'))
    df = feature_engineering(df)
    
    # Train and evaluate the model
    train_and_evaluate_model(df)
