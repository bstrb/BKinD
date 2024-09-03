# predict_dynamical_effects.py

def predict_dynamical_effects(model, df):
    # Ensure that df contains only the necessary columns for prediction
    features = df[['cbi', 'snr']]  # Assuming 'cbi' and 'snr' are the features used for training
    
    # Predict probabilities using the trained model
    predicted_probabilities = model.predict_proba(features)[:, 1]  # Probability of the positive class
    
    # Add the predicted probabilities to the DataFrame
    df['predicted_prob'] = predicted_probabilities
    
    return df
