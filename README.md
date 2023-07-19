<div align="center">
<h1>Reconstructing Missing Joints in 3D Human Motion with Temporal-Structural Awareness Graph Neural Network</h1>
</div>

<div align="center">
<img src=https://github.com/minhduc1122002/TA-WLS/assets/78080278/0fe35035-42af-48ec-929b-d702fee1175e>
</div>

## Install dependencies:

```
pip install -r requirements.txt
```
 
## Get the data

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

## Training
The arguments for running the code are defined in [parser.py](utils/parser.py)

#### TA-WLS
```bash
python main_h36_3d.py --data_path /PATH/TO/DATASET --model_name tawls
```
#### TA-GCN
```bash
python main_h36_3d.py --data_path /PATH/TO/DATASET --model_name tagcn
```
#### ST-GCN
```bash
python main_h36_3d.py --data_path /PATH/TO/DATASET --model_name stgcn
```

After Training the model checkpoint is saved to ./ directory

## Testing

```bash
python main_h36_3d.py --data_path /PATH/TO/DATASET --model_name tawls --model_path PATH/TO/CHECKPOINT
```

 
 ### Acknowledgments
 
 Some of our code was adapted from [HisRepsItself](https://github.com/wei-mao-2019/HisRepItself) by [Wei Mao](https://github.com/wei-mao-2019).
