# Emotion_Recognition
In IEMOCAP dataset, the two most important emotional cues are speech and face. In this repo, the emotion recognition task using the state-of-the-art ML technique are described.

## Step 1: Data cleaning and preprocessing:

Most of the data cleaning and preprocessing are similar to my another work in [here](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/Combine_audiovisual_data.py) and [here](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/window_based_reformation.py). Therefore, the detailed description is not provided. However, a simple discussion is as follows:

__step 1__: extract the speech features using this [code](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/audio_feat_extract.praat). Deal with the missing values of facial/video features using the preprocessing [code](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/Combine_audiovisual_data.py). 

```
Speech features: pitch, energy, MFCC-12 coefficients, MFB-27 coefficients
Facial features: 46 facial region landmark features on following locations of the face:
chin, forehead, cheek, upper eyebrow, eyebrow, mouth
```
__step 2__: Create statistical features from the framewise-extracted speech and facial features. Statistical features are:

```
1. Mean, Standard Deviation, 1st and 3rd Quantile, Interquantile range of Pitch, 12 MFCCC coeffs, 27 MFB coeffs, Energy values.
2. Mean, Standard Deviation, 1st and 3rd Quantile, Interquantile range of facial landmark features. 
```
There will be a total of 895 feature and for the emotion recognition task, we choose four emotions namely *Anger, Happy, Neutral and Sad*. The features can be created by setting ```window_type = 'static'``` in the ```process_data``` function of [window_based_reformation.py](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/window_based_reformation.py) code and by setting ```step = 0``` in ```creating_dataset``` function of [Utt_Fore_Data_Prep.py](https://github.com/sadat1971/Emotion-Forecasting/blob/master/Codes/Utt_Fore_Data_Prep.py) code.



