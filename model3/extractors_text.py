import pandas as pd
import re
try:
    from .extractors_base import BaseExtractor
except ImportError:
    from extractors_base import BaseExtractor

class TextStatsExtractor(BaseExtractor):
    
    def __init__(self, include_generic=True):
        self.include_generic = include_generic

    def _extract_stats(self, series: pd.Series) -> pd.DataFrame:
        df_feats = pd.DataFrame(index=series.index)
        
        post_length = series.str.len().fillna(0)
        
        if self.include_generic:
            df_feats['feat_post_length'] = post_length
            df_feats['feat_num_words'] = series.str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        df_feats['feat_code_block_count'] = series.str.count(r'<code>(?!.*<code>)').fillna(0)
        
        url_pattern = r'href=[\'"]?([^\'" >]+)'
        df_feats['feat_url_count'] = series.str.findall(url_pattern).apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        df_feats['feat_punct_count'] = series.str.count(r'[{}();,.]').fillna(0)
        df_feats['feat_punct_ratio'] = (df_feats['feat_punct_count'] / (post_length + 1)).fillna(0)
        
        return df_feats

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        return self._extract_stats(series)

    def transform(self, series: pd.Series) -> pd.DataFrame:
        return self._extract_stats(series)
