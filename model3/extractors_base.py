from abc import ABC, abstractmethod
import pandas as pd

class BaseExtractor(ABC):
    
    @abstractmethod
    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def transform(self, series: pd.Series) -> pd.DataFrame:
        pass
