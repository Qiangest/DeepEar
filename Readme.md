# DeepEar Dataset
Dataset and demo code of DeepEar.


## Binaural audio files

Users can build their own project with below raw data ([download](https://connectpolyu-my.sharepoint.com/:f:/g/personal/19044952r_connect_polyu_hk/EoJJZQvE371Hn_e5UMgDfYYBgehcFqZXR90z2UDCnJM8jQ?e=cYAHTs)).

```
Anechoic.7z: Audio data for the anechoic environment. 
|-----simulatedAudioTrain: audio for training.
|     |--------label_1.mat: ground truth labels of audio with one sound source.    SourceVector: source number. Radius: source distance. Sector: source sector. AoA: source direction.
|     |--------label_2.mat: ground truth labels of audio with two sound sources.
|     |--------label_3.mat: ground truth labels of audio with three sound sources.
|     |--------simulatedAudio_1_1500-1.wav: audio data. Sampling frequency: 16 KHz. Duration: 20s. Name: simulatedAudio_sourceNumber_numberOfFilesForThisSourceNumber-index.wav
|-----simulatedAudioTest: audio for testing.

Reverberant.7z: Audio data for the reverberant environment.
```

## Files
- DeepEar.py: Demo code of DeepEar.
  - TensorFlow 2.5.0
  - [mat73](https://pypi.org/project/mat73/) is required.
- DeepEar_weights.h5: Pretrained model weights. 
- TrainData.mat: Extracted features for training data.
  - Matlab v7.3 version file. 
  - Four column are gammatone coefficients of left ear, right ear, cross-correlation, and ground truth labels. Please refer to our paper for more details.
- TestData.mat: Extracted features for testing data.

## Label format

1 X 48 vector. Please refer to our paper for more details.

            [1]: binary sectors. [2]: AoA (0~1). [3-7]: one-hot distance
            [8]: binary sectors. [9]: AoA (0~1). [10-14]: one-hot distance
            [15]: binary sectors. [16]: AoA (0~1). [17-21]: one-hot distance
            [22]: binary sectors. [23]: AoA (0~1). [24-28]: one-hot distance
            [29]: binary sectors. [30]: AoA (0~1). [31-35]: one-hot distance
            [36]: binary sectors. [37]: AoA (0~1). [38-42]: one-hot distance
            [43]: binary sectors. [44]: AoA (0~1). [45-49]: one-hot distance
            [50]: binary sectors. [51]: AoA (0~1). [52-56]: one-hot distance



## Credits and License

The dataset and code are provided by Qiang Yang under the guidance of Prof. Yuanqing Zheng of The Hong Kong Polytechnic University (PolyU). They are licensed under CC-BY-NC.


For any questions, you may contact: qiang {dot} yang {at} connect {dot} polyu {dot} hk.

## Cite this work

```
@inproceedings{yang22DeepEar,
title={DeepEar: Sound Localization with Binaural Microphones},
author={Yang, Qiang and Zheng, Yuanqing},
booktitle = {Proceedings of the International Conference on Computer Communications (INFOCOM~'22)},
year = {2022},
publisher={IEEE}
}
```

 <img src="by-nc.png" width="15%">  

