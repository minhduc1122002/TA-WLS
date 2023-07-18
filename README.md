<div align="center">
<h1>Reconstructing Missing Joints in 3D Human Motion with Temporal-Structural Awareness Graph Neural Network</h1>
</div>

<div align="center">
<!-- <img src=https://user-images.githubusercontent.com/78080278/230953331-295ea805-1460-4dc2-98f5-7a8cb21616cc.png> -->
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

### Train with different missing mode
The arguments for running the code are defined in [parser.py](utils/parser.py)
  ```bash
  python main_h36_3d.py --data_dir /PATH/TO/DATASET --missing_mode random
  ```
 
 ### Acknowledgments
 
 Some of our code was adapted from [HisRepsItself](https://github.com/wei-mao-2019/HisRepItself) by [Wei Mao](https://github.com/wei-mao-2019).
