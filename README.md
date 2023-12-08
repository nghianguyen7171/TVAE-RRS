# Temporal-Variational-Autoencoder-Model-for-In-hospital-Clinical-Emergency-Prediction
Source for the paper "Temporal  Variational Autoencoder Model for In-hospital-Clinical-Emergency-Prediction"

# Abstract
Early recognition of clinical deterioration plays a pivotal role in a Rapid Response System (RRS), and it has been a crucial step in reducing inpatient morbidity and mortality. Traditional Early Warning Scores (EWS) and Deep Early Warning Scores (DEWS) for identifying patients at risk are still limited because of the challenges of imbalanced multivariate temporal data. Typical issues leading to their limitations are low sensitivity, high late alarm rate, and lack of interpretability; this has made the system face difficulty in being deployed in clinical settings. This study develops an early warning system based on Temporal Variational Autoencoder (TVAE) and a window interval learning framework, which uses the latent space features generated from the input multivariate temporal features, to learn the temporal dependence distribution between the target labels (clinical deterioration probability). Implementing the target information in the Fully Connected Network (FCN) architect of the decoder with a loss function assists in addressing the imbalance problem and improving the performance of the time series classification task. 

# Method


![BSPC-Method](https://github.com/nghianguyen7171/Temporal-Variational-Autoencoder-Model-for-In-hospital-Clinical-Emergency-Prediction/assets/35287087/40c5ff71-65b0-46d9-ab97-3d94e8243fc5)
