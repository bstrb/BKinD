# Define function for combined metric calculation using product of (1 + metric) ^ (metric weight)
def calculate_combined_metric(index, all_metrics, metric_weights):
    combined_metric = 1  # Start with 1 for multiplication
    for metric, weight in metric_weights.items():
        metric_value = all_metrics[metric][index]
        combined_metric *= (1 + metric_value) ** weight
    return combined_metric