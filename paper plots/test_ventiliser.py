from ventiliser.GeneralPipeline import GeneralPipeline
import multiprocessing
import pandas as pd
print(pd.__version__)
import numpy as np

# Print the numpy version
print(np.__version__)
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    multiprocessing.freeze_support()
    pipeline = GeneralPipeline()
    pipeline.configure() # For information on parameters you can configure see docs
    path=r"C:\Users\silas\Master Project\Meerkat Vital Sign Monitoring\dataset\pipeline intermediate\pipeline intermediate.csv"
    pipeline.load_data(path, [0,1,2]) # [0,1,2] refers to the columns in your data file corresponding to time, pressure, flow
    pipeline.process() # You can suppress log and output files by setting them false. See docs for more information