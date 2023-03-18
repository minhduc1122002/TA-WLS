<div align="center">
<h1>Predicting Missing Markers In Human Motion Using Graph Convolutional And Temporal Attention Network</h1>
</div>

 ### Install dependencies:
```
 $ pip install -r requirements.txt
```
 
### Get the data

[Human3.6m](http://vision.imar.ro/human3.6m/description.php) in exponential map can be downloaded from [here](http://www.cs.stanford.edu/people/ashesh/h3.6m.zip).
 
Directory structure: 
```shell script
H3.6m
|-- S1
|-- S5
|-- S6
|-- ...
`-- S11
```

[AMASS](https://amass.is.tue.mpg.de/en) from their official website. (In Progress)
 

Directory structure:
```shell script
amass
|-- ACCAD
|-- BioMotionLab_NTroje
|-- CMU
|-- ...
`-- Transitions_mocap
```
[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) from their official website. (In Progress)

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```
Put the all downloaded datasets in ../datasets directory.

### Train
The arguments for running the code are defined in [parser.py](utils/parser.py)
  ```bash
  python main_h36_3d.py
  ```
 
### Test
To test on the pretrained model, we have used the following commands:
 ```bash
 python main_h36_3d.py --mode test
  ```
 
 ### Acknowledgments
 
 Some of our code was adapted from [HisRepsItself](https://github.com/wei-mao-2019/HisRepItself) by [Wei Mao](https://github.com/wei-mao-2019).
