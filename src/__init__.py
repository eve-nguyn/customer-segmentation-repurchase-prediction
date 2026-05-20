"""
src package
-----------
Hybrid Customer Segmentation and Churn Prediction Framework
using RFMT Features and Machine Learning.

Modules
-------
data_loader          : load & clean UCI Online Retail II CSVs
outlier_utils        : IQR detection, percentile capping, visualisation
feature_engineering  : RFMT computation, temporal split, EDA
clustering           : K-Means, Fuzzy C-Means, Spectral, MeanShift
classification       : cluster-based & global churn classification
cluster_profiling    : RFMT cluster profiling charts
main                 : end-to-end pipeline runner
"""
