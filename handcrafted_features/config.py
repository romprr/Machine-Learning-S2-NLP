import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDCRAFTED_DIR = os.path.join(BASE_DIR, "handcrafted_features")

# Input
RAW_DATA_PATH = os.path.join(HANDCRAFTED_DIR, "stack-overflow-data.csv")

# Output
FEATURES_OUTPUT_PATH = os.path.join(HANDCRAFTED_DIR, "handcrafted_features.csv")
METADATA_OUTPUT_PATH = os.path.join(HANDCRAFTED_DIR, "features_metadata.json")
EXPLORATION_REPORT_PATH = os.path.join(HANDCRAFTED_DIR, "exploration_report.md")
DISCOVERED_FEATURES_PATH = os.path.join(HANDCRAFTED_DIR, "discovered_candidates.json")
