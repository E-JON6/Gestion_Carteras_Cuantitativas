"""
Created on Tue Mar 24 12:48:39 2026

@author: andre
"""
import numpy as np
import pandas as pd

def select_top_etfs(scores, top_n=3):
    clean_scores = scores.dropna().sort_values(ascending=False)
    return list(clean_scores.head(top_n).index)

