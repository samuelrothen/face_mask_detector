import pandas as pd
import matplotlib.pyplot as plt


hist_file='../models/mask_detection_model_hist_df.pkl'
df_hist=pd.read_pickle(hist_file)
df_hist.plot()