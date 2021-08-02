# D3_map_segmentation
Paper: A Novel Framework with Weighted Decision Map Based on Convolutional Neural Network for Cardiac MR Segmentation(UnderReview)

Data_aug.py Data augmentation
Get_map.py Get decision map
Get_weighted_map.py Get weighted map
Loss_function.py Show loss function
Map_Network.py Contains the network structure of all learning decision maps used in the paper
Metrics.py Contains all evaluation indicators used in the paper
Network.py Contains all network structures used in the paper
Train_2017ACDC.py train segmentation network
Train_map.py train decision map

steps:
1.Get_map.py
2.Train_map.py 
3.Predict_2017ACDC_map.py
4.Get_weighted_map.py
5.Train_2017ACDC.py
6.Predict_2017ACDC.py
7. ACDC test: https://acdc.creatis.insa-lyon.fr

MS-CMRSeg 2019, MyoPS 2020, ACDC 2017: The training method of the three data sets is the same.
