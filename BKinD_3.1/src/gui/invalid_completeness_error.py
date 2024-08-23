# invalid_completeness_error.py

class InvalidCompletenessError(Exception):
    def __init__(self, start_completeness, target_completeness):
        self.start_completeness = start_completeness
        self.target_completeness = target_completeness
        super().__init__(f"Invalid target completeness: {target_completeness}. Target completeness must be less than start completeness: {start_completeness}.")
    