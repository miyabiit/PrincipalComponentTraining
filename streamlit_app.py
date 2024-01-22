import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go

st.title("主成分分析 Wine価格")

dataset = load_wine()
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["target"] = dataset.target

if st.checkbox("Wineデータセットの詳細"):
    dataset.DESCR
if st.checkbox("Wineデータセット閲覧"):
    df

# standard
x = df.drop(columns = ["target"])
y = df["target"]

sscaler = StandardScaler()
x_std = sscaler.fit_transform(x)

st.sidebar.markdown(
        r"""
        ### number of Pricipal Factors
        """
        )
num_pca = st.sidebar.number_input(
        "Minimum Value",
        value = 3,
        step = 1,
        min_value = 3
        )

pca = PCA(n_components = num_pca)
x_pca = pca.fit_transform(x_std)

# show
st.sidebar.markdown(
        r"""
        ### Select components to plot
        """
        )
idx_x = st.sidebar.selectbox("X axis:", np.arange(1,num_pca+1),0)
idx_y = st.sidebar.selectbox("Y axis:", np.arange(1,num_pca+1),1)
idx_z = st.sidebar.selectbox("Z axis:", np.arange(1,num_pca+1),2)

x_lbl = f"PCA{idx_x}"
x_plot = x_pca[:,idx_x - 1]

y_lbl = f"PCA{idx_y}"
y_plot = x_pca[:,idx_y - 1]

z_lbl = f"PCA{idx_z}"
z_plot = x_pca[:,idx_z - 1]

