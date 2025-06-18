# AI-ML-Audio-Processing
# 🎧 Emotion Recognition from Speech - Streamlit App

### app-link = {https://ai-ml-audio-processing-bkkb7cu5yivrcbfewxntft.streamlit.app/}

This project is a full machine learning pipeline to classify human emotions from speech using MFCC audio features. The trained model is deployed using Streamlit Cloud, allowing users to upload `.wav` or `.mp3` files and get real-time emotion predictions.

---

## Project Description

We aim to classify emotional states (such as **happy**, **sad**, **angry**, etc.) using audio-only data. The app takes a speech audio file as input, extracts its MFCC features, and uses a trained XGBoost model to predict the speaker’s emotional state.

- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Modalities used**: Audio (speech/song)
- **Classes**: Calm, Happy, Sad, Angry, Fearful, Disgust
- **MFCC(Mel Freuqency Cepstral Coefficient**: MFCC is used to convert the audio into features that can be used to extract the information from the audio which will then processed through XG Boost                                                 algorithm.MFCC converted the audio time signal to frequency signal through fourier transform.Then mel scaling is applied to frequency signal and then
                                               taking the logarithm as the humans are sensitive to lower frequency signals as compared to high frequency signals so it shrinks the high frequancy                                                    signal but the lower frequency signals remain linear.Then finally the DCT is applied and the signal is converted to coefficients.So we obtain mfcc 
                                               coefficients (13- 40).

##Accuracy Metrics
-**Overall Accuracy**: ~78% on the test set

-**Macro F1-score**: ~0.76

-**Confusion Matrix showed best performance for**:

Neutral and Calm (high precision/recall)

Some confusion between Sad and Fearful (expected due to acoustic similarity)

