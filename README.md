# Temporal-Variational-Autoencoder-Model-for-In-hospital-Clinical-Emergency-Prediction
Source for the paper "Temporal  Variational Autoencoder Model for In-hospital-Clinical-Emergency-Prediction"

# Abstract
Early recognition of clinical deterioration plays a pivotal role in a Rapid Response System (RRS), and it has been a crucial step in reducing inpatient morbidity and mortality. Traditional Early Warning Scores (EWS) and Deep Early Warning Scores (DEWS) for identifying patients at risk are still limited because of the challenges of imbalanced multivariate temporal data. Typical issues leading to their limitations are low sensitivity, high late alarm rate, and lack of interpretability; this has made the system face difficulty in being deployed in clinical settings. This study develops an early warning system based on Temporal Variational Autoencoder (TVAE) and a window interval learning framework, which uses the latent space features generated from the input multivariate temporal features, to learn the temporal dependence distribution between the target labels (clinical deterioration probability). Implementing the target information in the Fully Connected Network (FCN) architect of the decoder with a loss function assists in addressing the imbalance problem and improving the performance of the time series classification task. Thus, we validated our proposed method on an in-house clinical dataset collected from Chonnam National University Hospitals (CNUH) and a public dataset from the University of Virginia (UV). Extensive trials with diverse validation methods, data sizes, and study cases demonstrate that our system outperforms existing methods, showcasing remarkable performance across usual criteria used to evaluate classification models, typically as the Area Under The Receiver Operating Characteristic Curve (AUROC) and the Area Under The Precision-Recall Curve (AUPRC). Our system also offers a significant reduction of late alarm issues on two datasets. The experimental process also demonstrates the stable performance of the proposed system under the limited number of data samples. 

# Method


![BSPC-Method](https://github.com/nghianguyen7171/Temporal-Variational-Autoencoder-Model-for-In-hospital-Clinical-Emergency-Prediction/assets/35287087/40c5ff71-65b0-46d9-ab97-3d94e8243fc5)


# Citation
+ BibTeX:
@article{Nguyen2025,
author = {Nguyen, Trong-nghia and Kim, Soo-hyung and Kho, Bo-gun and Do, Nhu-tai},
doi = {10.1016/j.bspc.2024.106975},
issn = {1746-8094},
journal = {Biomedical Signal Processing and Control},
keywords = {Clinical deterioration,Rapid response system,Deep learning,Clinical medical signal,clinical deterioration,rapid response system},
number = {PC},
pages = {106975},
publisher = {Elsevier Ltd},
title = {{Biomedical Signal Processing and Control Temporal variational autoencoder model for in-hospital clinical emergency prediction}},
url = {https://doi.org/10.1016/j.bspc.2024.106975},
volume = {100},
year = {2025}

+ Text:
Trong-Nghia Nguyen, Soo-Hyung Kim, Bo-Gun Kho, Nhu-Tai Do, Ngumimi-Karen Iyortsuun, Guee-Sang Lee, Hyung-Jeong Yang,
"Temporal variational autoencoder model for in-hospital clinical emergency prediction",
Biomedical Signal Processing and Control,
Volume 100, Part C,
2025,
106975,
ISSN 1746-8094,
https://doi.org/10.1016/j.bspc.2024.106975.

