<<<<<<< Updated upstream
# PointNetGPD
The original codes are from https://github.com/lianghongzhuo/PointNetGPD.  
Since I used other training data generalized by Haoshu Fang and Chenxi Wang, I made a little bit of adjustment in the input of the network.  
The vital parameter to change is the input path.  
The data to train should be like this:   

DATA  
|--train1  
|--train2  
|--train3  
&emsp;&emsp;|--labels_train3.npy  
&emsp;&emsp;|--cloud  
&emsp;&emsp;&emsp;&emsp;|--000001.ply   
&emsp;&emsp;&emsp;&emsp;|--000002.ply    
  
=======
# PointNetGPD-revised

## Introduction 

## Files

## Requirements
In order to use our codes to train the model, you need the Minkowski Engine dataset generalized by Haoshu Fang and Chenxi Wang, and the file structure of the dataset should be as follows.
```bash
gpd_data
├── train1
|   ├── image
|   |   ├── 000000.jpg
|   |   ├── 000001.jpg
|   |   └── ...
|   └── labels_train1.npy
├── train2
|   └── ...
├── train3
|   └── ...
├── train4
|   └── ...
└── test_seen
    └── ...
```
## Usages

## References

Training 250w+ data
>>>>>>> Stashed changes
