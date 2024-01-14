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
df.target = dataset.target

if st.checkbox("Wineデータセットの詳細"):
    dataset.DESCR
if st.checkbox("Wineデータセット閲覧"):
    df

