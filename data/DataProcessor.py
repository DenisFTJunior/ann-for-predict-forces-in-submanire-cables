import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    x_train, x_test, y_train, y_test, x_val, y_val  = None, None, None, None, None, None
    col = None
    def __init__(self, data, transformer, col: str = None):
        self.data =  transformer(data)
        self.col = col

    def preprocess(self):
        delta_col = 'Delta yaw neutro'
        y_col = self.col if self.col else 'Tensão % nas linhas'
        base_cols = [col for col in self.data.columns if (col.startswith('R') or col.startswith('P')) and col != 'Peso do sistema']
        all_cols = base_cols + [delta_col, y_col]
        self.data = self.data[self.data[delta_col] < 3]
        self.data =  self.data[self.data[y_col] < 100]
        x = self.data[all_cols].copy()
        y = x.pop(y_col).values
        normalized_x = StandardScaler().fit(x).transform(x)
        return normalized_x, y
    
    def preprocess_multiple_outputs(self):
        delta_col = 'Delta yaw neutro'
        y_cols = ['Tensão % nas linhas', '% de carga nas ancoras', 'Offset % a partir do neutro']  # Example of multiple output columns
        base_cols = [col for col in self.data.columns if (col.startswith('R') or col.startswith('P')) and col != 'Peso do sistema']
        all_cols = base_cols + [delta_col] + y_cols
        self.data = self.data[self.data[delta_col] < 3]
        for y_col in y_cols:
            self.data = self.data[self.data[y_col] < 100]
        x = self.data[all_cols].copy()
        y = x[y_cols].values
        x = x.drop(columns=y_cols)
        normalized_x = StandardScaler().fit(x).transform(x)
        return normalized_x, y

    def split_data(self, test_size=0.2, val_size=0.2, random_state=4, multiple_outputs=False):
        x, y = self.preprocess() if not multiple_outputs else self.preprocess_multiple_outputs()
        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        val_relative_size = val_size / (1 - test_size)
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=val_relative_size, random_state=random_state)
        # Assign raw arrays to instance attributes
        self.x_train, self.x_val, self.x_test = x_train, x_val, x_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test


    