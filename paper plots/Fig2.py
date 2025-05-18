import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\meerkat_analysis_pipeline")

import SummaryStatistics
# Define folder of the dataset
database_path = r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset"

SummaryStatistics.ground_truth_characterisation(database_path)