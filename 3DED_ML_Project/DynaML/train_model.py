# train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def train_model(df):
    # Define features
    X = df[['cbi', 'snr']]

    # Define target: assume dynamical effects if cbi < -1 and snr > median(snr)
    snr_median = np.median(df['snr'])
    y = (df['cbi'] < -1) & (df['snr'] > snr_median)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    return model
