import pandas as pd
import re
import os
import json
from .extractors_base import BaseExtractor

try:
    from .config import DISCOVERED_FEATURES_PATH
except ImportError:
    from config import DISCOVERED_FEATURES_PATH

class RegexFeatureExtractor(BaseExtractor):
    
    def __init__(self, use_auto_discovered=True, max_auto_per_tag=10):
        self.patterns = {
            # c++
            'feat_cpp_std': r'\bstd::',                     
            'feat_cpp_cout': r'\bcout\b',                   
            'feat_cpp_vector': r'\bvector\b',               
            
            # java
            'feat_java_println': r'\bSystem\.out\.println\b', 
            'feat_java_main': r'public\s+static\s+void\s+main',
            'feat_java_import': r'import\s+java\.',
            
            # javascript
            'feat_js_getelem': r'\bdocument\.getElementById\b',
            
            # php
            'feat_php_echo': r'\becho\b',                   
            'feat_php_superglobal': r'\$_\w+\[',
            
            # python
            'feat_py_def': r'def\s+\w+\s*\(',
            
            # ruby-on-rails
            'feat_ruby_activerecord': r'ActiveRecord',
            'feat_ruby_hasmany': r'has_many',
            'feat_ruby_rails': r'ruby\s+on\s+rails',
            
            # objective-c
            'feat_objc_interface': r'@interface',
            'feat_objc_impl': r'@implementation',
            'feat_objc_nsstring': r'NSString\s*\*?',
            
            # angularjs
            'feat_ang_ngapp': r'ng-app',
            'feat_ang_scope': r'\$scope',
            'feat_ang_model': r'ng-model',
            'feat_ang_controller': r'ng-controller',
            'feat_ang_http': r'\$http',
            'feat_ang_module': r'angular\.module',
            
            # c
            'feat_c_intmain': r'int\s+main\s*\(\s*\)',
            'feat_c_printf': r'printf\s*\(',
            
            # jquery
            'feat_jquery_docready': r'\$\(\s*document\s*\)\.\s*ready',
            
            # css
            'feat_css_bgcolor': r'background-color:',
            
            # asp.net
            'feat_aspnet_sysweb': r'System\.Web',
            
            # c#
            'feat_cs_writeline': r'\bConsole\.WriteLine\b'
        }

        if use_auto_discovered and os.path.exists(DISCOVERED_FEATURES_PATH):
            with open(DISCOVERED_FEATURES_PATH, "r", encoding="utf-8") as f:
                auto_feats = json.load(f)
            
            for tag, candidates in auto_feats.items():
                for i, cand in enumerate(candidates[:max_auto_per_tag]):
                    clean_tag = tag.replace("-", "_").replace(".", "dot").replace("+", "p")
                    feat_name = f"feat_auto_{clean_tag}_{i}"
                    self.patterns[feat_name] = self._build_dynamic_regex(cand['token'])

    def _build_dynamic_regex(self, token):
        parts = token.split()
        escaped = [re.escape(p) for p in parts]
        regex_str = escaped[0]
        
        for i in range(1, len(parts)):
            prev = parts[i-1]
            curr = parts[i]
            
            if prev[-1].isalnum() and curr[0].isalnum():
                regex_str += r'\s+'
            else:
                regex_str += r'\s*'
            regex_str += escaped[i]
            
        if parts[0][0].isalnum():
            regex_str = r'\b' + regex_str
        if parts[-1][-1].isalnum():
            regex_str = regex_str + r'\b'
            
        return regex_str

    def _apply_patterns(self, series: pd.Series) -> pd.DataFrame:
        df_feats = pd.DataFrame(index=series.index)
        for feat_name, pattern in self.patterns.items():
            df_feats[feat_name] = series.str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False).astype(int)
        return df_feats

    def fit_transform(self, series: pd.Series) -> pd.DataFrame:
        return self._apply_patterns(series)
        
    def transform(self, series: pd.Series) -> pd.DataFrame:
        return self._apply_patterns(series)
