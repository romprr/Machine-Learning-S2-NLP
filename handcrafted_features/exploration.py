import pandas as pd
import re
import os

try:
    from .config import RAW_DATA_PATH, EXPLORATION_REPORT_PATH
except ImportError:
    from config import RAW_DATA_PATH, EXPLORATION_REPORT_PATH

class ExplorationReporter:
    def __init__(self, output_path):
        self.output_path = output_path
        self.report_lines = []
        self.kept_features = []
        self.discarded_features = []
    
    def log(self, text=""):
        print(text)
        self.report_lines.append(text)
        
    def add_feature_eval(self, pattern, tag, precision, coverage, threshold_cov=200, threshold_prec=0.5):
        is_kept = coverage >= threshold_cov and precision >= threshold_prec
        status = "KEPT" if is_kept else "DISCARDED"
        eval_str = f"- **{pattern}** (Target: `{tag}`) - Precision: **{precision:.2f}**, Coverage: **{coverage}** -> {status}"
        self.log(eval_str)
        
        feature_data = {'pattern': pattern, 'tag': tag, 'precision': precision, 'coverage': coverage}
        if is_kept:
            self.kept_features.append(feature_data)
        else:
            self.discarded_features.append(feature_data)

    def write_report(self):
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("# Handcrafted Features Exploration Report\n\n")
            f.write("## Criteria for Acceptance\n")
            f.write("- Coverage: >= 200\n")
            f.write("- Precision: >= 50%\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total Kept Features: {len(self.kept_features)}\n")
            f.write(f"- Total Discarded Features: {len(self.discarded_features)}\n\n")
            
            f.write("## Details of Accepted Features\n")
            for f_res in self.kept_features:
                f.write(f"- `{f_res['pattern']}` (Target: {f_res['tag']}) [Coverage: {f_res['coverage']} | Precision: {f_res['precision']:.2f}]\n")
            f.write("\n---\n")
            f.write("## Feature Evaluation Log\n\n")
            f.write("\n".join(self.report_lines))
        self.log(f"\n[INFO] Full exploration report successfully saved to: {self.output_path}")

def run_basic_stats(df, reporter):
    reporter.log("### 1. Basic Dataset Statistics")
    reporter.log(f"- **Total Rows:** {df.shape[0]}")
    reporter.log(f"- **Columns:** {', '.join(df.columns.tolist())}\n")

def evaluate_dictionary(df, patterns_dict, reporter):
    for tag, pats in patterns_dict.items():
        reporter.log(f"#### Target tag: `{tag}`")
        tag_mask = df['tags'] == tag
        for pat in pats:
            matches = df['post'].str.contains(pat, flags=re.IGNORECASE, regex=True, na=False)
            total_matches = matches.sum()
            precision = 0
            coverage = (matches & tag_mask).sum()
            if total_matches > 0:
                precision = coverage / total_matches
            
            reporter.add_feature_eval(pat, tag, precision, coverage)
        reporter.log("")

def analyze_regex_patterns(df, reporter):
    reporter.log("### 2. Standard Regex & Keyword Patterns Evaluation\n")
    
    patterns = {
        'python': [r'\bdef\b', r'\bimport\b', r'\bprint\b', r'\bself\b', r'\bdict\b', r'\blist\b', r'\bpandas\b', r'\bmatplotlib\b'],
        'c++': [r'#include\b', r'\bstd::', r'\bcout\b', r'\bvector\b'],
        'java': [r'\bpublic\s+class\b', r'\bString\[\]\s+args\b', r'\bSystem\.out\.println\b', r'\bArrayList\b', r'\bprintStackTrace\b'],
        'javascript': [r'\bvar\b', r'\bfunction\s*\(', r'\bconsole\.log\b', r'\bdocument\.getElementById\b'],
        'php': [r'\$_\w+', r'\becho\b', r'\bpreg_match\b', r'\bmysqli\b'],
        'html': [r'<\s*div\b', r'<\s*span\b', r'<\s*a\b', r'<\s*html\b'],
        'sql': [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bJOIN\b', r'\bGROUP BY\b'],
        'c#': [r'\bConsole\.WriteLine\b', r'\busing\s+System', r'\bnamespace\b', r'\bIList\b']
    }
    evaluate_dictionary(df, patterns, reporter)

def analyze_urls_and_domains(df, reporter):
    reporter.log("### 3. Domain & URL Mentions Analysis\n")
    domains_to_check = {
        'php': ['php.net'],
        'java': ['docs.oracle.com', 'spring.io'], 
        'c#': ['docs.microsoft.com'],
        'javascript': ['developer.mozilla.org']
    }
    evaluate_dictionary(df, domains_to_check, reporter)

def analyze_robust_combinations(df, reporter):
    reporter.log("### 4. Robust API & Multi-Word phrase Evaluation\n")
    combos = {
        'ruby-on-rails': [r'def\s+index', r'class\s+\w+\s*<\s*ApplicationController', r'ruby\s+on\s+rails', r'rails\s+generate', r'gem\s+[\'"]\w+[\'"]', r'ActiveRecord', r'has_many'],
        'ios': [r'import\s+UIKit', r'UIViewController', r'viewDidLoad', r'override\s+func'],
        'objective-c': [r'#import\s+<', r'@interface', r'@implementation', r'#import\s+[\'"]\w+', r'\[self\s+\w+\]', r'NSString\s*\*', r'NSArray\s*\*', r'NSDictionary\s*\*'],
        'angularjs': [r'ng-app', r'ng-model', r'ng-controller', r'\$scope', r'\$http', r'angular\.module'],
        'iphone': [r'NSObject', r'CGRectMake', r'IBOutlet', r'IBAction'],
        'c': [r'#include\s+<stdio\.h>', r'#include\s+<stdlib\.h>', r'int\s+main\s*\(\s*\)', r'printf\s*\('],
        'mysql': [r'INSERT\s+INTO\s+\w+', r'AUTO_INCREMENT', r'mysql_query', r'phpMyAdmin'],
        'jquery': [r'\$\(\s*[\'"]#\w+', r'\$\(\s*function\s*\(', r'\$\.\s*ajax', r'\$\(\s*document\s*\)\.\s*ready'],
        'css': [r'\{\s*margin:', r'\{\s*padding:', r'background-color:', r'\.css\s*\('],
        'sql': [r'SQL\s+Server', r'FOREIGN\s+KEY', r'PRIMARY\s+KEY', r'VARCHAR\('],
        'c#': [r'public\s+class', r'using\s+System', r'static\s+void\s+Main', r'Page_Load', r'ToString\(\)', r'Convert\.ToInt32'],
        'asp.net': [r'asp:Button', r'asp:Label', r'Page_Load', r'runat="server"', r'System\.Web', r'asp:GridView']
    }
    evaluate_dictionary(df, combos, reporter)

def main():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Dataset not found at: {RAW_DATA_PATH}")
        return
        
    print(f"Loading data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    reporter = ExplorationReporter(EXPLORATION_REPORT_PATH)
    
    run_basic_stats(df, reporter)
    analyze_regex_patterns(df, reporter)
    analyze_urls_and_domains(df, reporter)
    analyze_robust_combinations(df, reporter)
    
    reporter.write_report()

if __name__ == "__main__":
    main()
