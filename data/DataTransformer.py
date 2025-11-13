import pandas as pd

class DataTransformer:
    
    @staticmethod
    def get_transformer( type = 'csv'):
        if type == 'csv':
            return pd.read_csv
        elif type == 'json':
            return pd.read_json
        elif type == 'excel':
            return pd.read_excel
        else:
            raise ValueError(f"Unsupported data type: {type}")