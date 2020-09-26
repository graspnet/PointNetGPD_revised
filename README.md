# PointNetGPD
The original codes are from https://github.com/lianghongzhuo/PointNetGPD.  
Since I used other training data generalized by Haoshu Fang and Chenxi Wang, I made a little bit of adjustment in the input of the network.  
The vital parameter to change is the input path.  
The data to train should be like this:   

DATA  
|--train1  
|--train2  
|--train3  
      |--labels_train3.npy  
      |--cloud  
            |--000001.ply  
            |--000002.ply  
  
