import pandas as pd
import matplotlib.pyplot as plt

# Quick Visualization of the Model Training History File

hist_file = '../models/mask_detection_model_hist_df.pkl'
df_hist = pd.read_pickle(hist_file)

fig = plt.figure('Model History')
fig.clf()
ax = plt.subplot(111)
df_hist.plot(ax=ax)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy / Loss')
