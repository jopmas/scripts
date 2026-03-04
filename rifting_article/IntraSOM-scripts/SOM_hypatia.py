from datetime import datetime, timedelta
start_datetime = datetime.now()
print(f"Current date and time: {start_datetime}")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
import scipy.stats as spstats
from random import sample,seed
from numpy.random import uniform
import pandas as pd
import json
import os
import sys
ncores = int(sys.argv[1]) #first argument
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] ## store default color cycle
seed(42)

# Intrasom
import intrasom
from intrasom.visualization import PlotFactory
from intrasom.clustering import ClusterFactory


df_file = 'KV3-SOM_points-CRUST.csv'
main_df = pd.read_csv(df_file,sep=';',index_col=0)
print("Data loaded!")

# filter the boundary effect
main_df = main_df.loc[main_df["x"]>=200]
main_df = main_df.loc[main_df["x"]<=1000]

main_df['sigma'] = main_df['sigma']/1e6 #Pa to MPa

#filtering Sigma
sigma_q005 = main_df['sigma'].quantile(0.005)
sigma_q500 = main_df['sigma'].quantile(0.5)
sigma_q995 = main_df['sigma'].quantile(0.995)

print(f"Q0.5: {sigma_q005:.3f}\nQ50: {sigma_q500:.3f}\nQ99.5: {sigma_q995:.3f}")
main_df['sigma'].loc[main_df['sigma']>sigma_q995] = sigma_q995
main_df['sigma'].loc[main_df['sigma']<sigma_q005] = sigma_q005
print(main_df['sigma'].describe())

print("Ploting histograms")
#strain
fig, axs = plt.subplots(1,2,figsize=(10,5))
fig.suptitle("Strain histogram")
sns.histplot(data=main_df,x="strain",ax=axs[0],stat="density")
sns.histplot(data=main_df,x="strain",ax=axs[1],stat="density",log_scale=True)
plt.savefig("strain_histogram.png",dpi=150)

#Strain rate
fig, axs = plt.subplots(1,2,figsize=(10,5))
fig.suptitle("Strain rate histogram")
sns.histplot(data=main_df,x="strain_rate",ax=axs[0],stat="density")
sns.histplot(data=main_df,x="strain_rate",ax=axs[1],stat="density",log_scale=True)
plt.savefig("strainrate_histogram.png",dpi=150)

#Viscosity
fig, axs = plt.subplots(1,2,figsize=(10,5))
fig.suptitle("viscosity histogram")
sns.histplot(data=main_df,x="viscosity",ax=axs[0],stat="density")
sns.histplot(data=main_df,x="viscosity",ax=axs[1],stat="density",log_scale=True)
plt.savefig("viscosity_histogram.png",dpi=150)

#temperature
fig, ax = plt.subplots(1,1,figsize=(5,5))
fig.suptitle("temperature histogram")
sns.histplot(data=main_df,x="temperature",ax=ax,stat="density")
plt.savefig("histogram_histogram.png",dpi=150)

#sigma
fig, ax = plt.subplots(1,1,figsize=(5,5))
fig.suptitle("sigma histogram")
sns.histplot(data=main_df,x="sigma",ax=ax,stat="density")
plt.savefig("sigma_histogram.png",dpi=150)

#---

df_som = main_df.loc[:,["x","z","time","density","temperature","sigma"]]
df_som['strain'] = np.log10(main_df['strain'])
df_som['strain_rate'] = np.log10(main_df['strain_rate'])
df_som['viscosity'] = np.log10(main_df['viscosity'])

corr_matrix = df_som.corr()
print(corr_matrix)
plt.figure(figsize=(20,10),dpi=300)
sns.heatmap(
    corr_matrix,
    annot=True,     # Mostra os números de correlação dentro de cada célula
    cmap='coolwarm',# Define o mapa de cores (vermelho-azul)
    fmt=".2f"       # Formata os números para 2 casas decimais
)
plt.savefig("corr_matrix.png",dpi=150)


print("starting training")
#training
seed(42)

comp_names = ["density","temperature","strain","strain_rate","sigma","viscosity"]
som_data = df_som.loc[:,comp_names].sample(frac=1,random_state=42)

som_name = 'KV3_0.5_None_70-400'
mapsize = None ## goes for Vesanto estimate
#mapsize = (40,40)

som_obj = intrasom.SOMFactory.build(som_data,
                                     mapsize= mapsize,
                                     mapshape='toroid',
                                     lattice='hexa',
                                     normalization='var',
                                     initialization='random',
                                     neighborhood='gaussian',
                                     training='batch',
                                     name=som_name,
                                     component_names=comp_names,
                                     unit_names = ["kg/m3","°C","NA","s-1","Pa","Pa.s"],
                                     sample_names=None,
                                     missing=False,
                                     dist_factor=2)

som_obj.train(n_job=ncores,
              train_rough_len=70,
              train_rough_radiusin=40,
              train_rough_radiusfin=15,
              train_finetune_len=400,
              train_finetune_radiusin=20,
              train_finetune_radiusfin=1,
              bootstrap=False
              )


## errors final
print('------------------------------')
print('Topographic = %.5f\nQuantization = %.5f'%(som_obj.topographic_error, som_obj.QE))


#plots
plot = PlotFactory(som_obj)
plot.multiple_component_plots(figsize = (12,3),
                              save=True)

plot.plot_umatrix(figsize = (13,2.5),
                  watermark_neurons=False,
                  samples_label=False,
                  hits=False,
                  title = "U-Matrix")


# Record the end time
end_datetime = datetime.now()

# Calculate the duration
duration = end_datetime - start_datetime
print(f"Duration: {duration}")
print(f"Duration in seconds: {duration.total_seconds()}")
