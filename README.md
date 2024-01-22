# hand_detection_evaluation
Evaluate DHP19 hand detection model running on chip on a subject on the DHP19 dataset

### Requirements
```bash
pip list | grep lava
pip list | grep nx
``` 
- lava-dl                   0.4.0
- lava-dnf                  0.1.4
- lava-loihi                0.5.0
- lava-nc                   0.8.0
- lava-optimization         0.3.0
- nxcore                    2.4.0
- on `ncl-edu.research.intel-research.net`

### DHP19 Data Subject 1
Download the data from <https://drive.switch.ch/index.php/s/8EvGwQmV7RgTR5o> and extract it in the `data/` folder.

```
cd data
wget https://drive.switch.ch/index.php/s/8EvGwQmV7RgTR5o/download
unzip download
rm -rf download data/ __MACOSX/
ls data/dhp19_samples/
```

The content of `data/dhp19_samples/` should look like this.
```
S1_session1_mov1_sample0.pt  S1_session1_mov2_sample2.pt  S1_session1_mov3_sample4.pt  S1_session1_mov5_sample1.pt  S1_session1_mov7_sample0.pt  S1_session1_mov8_sample0.pt
S1_session1_mov1_sample1.pt  S1_session1_mov2_sample3.pt  S1_session1_mov3_sample5.pt  S1_session1_mov5_sample2.pt  S1_session1_mov7_sample1.pt  
...
S1_session1_mov8_sample4.pt  S1_session2_mov1_sample52.pt  S1_session2_mov2_sample8.pt   S1_session2_mov3_sample54.pt  S1_session2_mov5_sample18.pt  S1_session2_mov6_sample7.pt   S1_session4_mov1_sample20.pt  S1_session4_mov1_sample67.pt  S1_session4_mov5_sample4.pt   S1_session5_mov7_sample8.pt
S1_session1_mov8_sample5.pt  S1_session2_mov1_sample53.pt  S1_session2_mov2_sample9.pt   S1_session2_mov3_sample55.pt  S1_session2_mov5_sample19.pt  S1_session2_mov6_sample8.pt   S1_session4_mov1_sample21.pt  S1_session4_mov1_sample68.pt  S1_session4_mov5_sample5.pt   S1_session5_mov7_sample9.pt
```

### DHP19 Model
`model/train/$model_dir_XYZ/train/` holds the current DHP19 hand detection SDNN model. See <https://github.com/neuromorphic-zhaw/sdnn-hand-detection-model> for the current model.