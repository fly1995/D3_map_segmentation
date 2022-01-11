# D3_map_segmentation
Paper: A Novel Framework with Weighted Decision Map Based on Convolutional Neural Network for Cardiac MR Segmentation

Data_aug.py Data augmentation
Get_map.py Get decision map
Get_weighted_map.py Get weighted map
Loss_function.py Show loss function
Map_Network.py Contains the network structure of all learning decision maps used in the paper
Metrics.py Contains all evaluation indicators used in the paper
Network.py Contains all network structures used in the paper
Train_2017ACDC.py train segmentation network
Train_map.py train decision map
(reviewers suggest deleting the experiments of ACDC in final version)

steps:
1.Get_map.py
2.Train_map.py 
3.Predict_2017ACDC_map.py
4.Get_weighted_map.py
5.Train_2017ACDC.py
6.Predict_2017ACDC.py
7. ACDC test: https://acdc.creatis.insa-lyon.fr

MS-CMRSeg 2019, MyoPS 2020: train and test(dada process: crop center area and data_aug)
External_validation_prediction.py
External_validation_prediction_map.py
External_validation_train.py
External_validation_train_map.py
compute_metrics.py

Requirements:
absl-py	0.10.0	
astor	0.8.1	
ca-certificates	2020.10.14	
certifi	2020.6.20	
chardet	3.0.4	
cloudpickle	1.6.0	
cycler	0.10.0	
debtcollector	2.2.0	
decorator	4.4.2	
gast	0.4.0	
gdcm	1.1	
grpcio	1.33.1	
h5py	2.10.0	
hausdorff	0.2.5	
idna	2.10	
image-utils	0.1.6	
imageio	2.9.0	
importlib-metadata	2.0.0	
iso8601	0.1.14	
jieba	0.42.1	
joblib	0.17.0	
keras	2.2.4	
keras-applications	1.0.8	
keras-preprocessing	1.1.2	
kiwisolver	1.2.0	
llvmlite	0.34.0	
markdown	3.3.3	
matplotlib	3.3.2	
medpy	0.4.0	
mock	4.0.2	
netaddr	0.8.0	
netifaces	0.10.9	
networkx	2.5	
nibabel	3.2.0	
numba	0.51.2	
numpy	1.19.2	
opencv-python	4.4.0.44	
openssl	1.1.1h	
oslo-i18n	5.0.1	
oslo-utils	4.8.0	
packaging	20.4	
pandas	1.2.0	
pbr	5.5.1	
pillow	8.0.1	
pip	20.2.4	
plotly	4.14.1	
progressbar	2.5	
progressbar2	3.53.1	
protobuf	3.13.0	
pydicom	2.1.2	
pydot	1.4.2	
pydot-ng	2.0.0	
pydotplus	2.0.2	
pylibjpeg	1.1.1	
pylibjpeg-libjpeg	1.1.0	
pylibjpeg-openjpeg	1.0.1	
pynrrd	0.4.2	
pyparsing	2.4.7	
python	3.7.9	
python-cephlibs	0.94.5.post1	
python-dateutil	2.8.1	
python-graphviz	0.16	
python-utils	2.4.0	
pytz	2020.5	
pywavelets	1.1.1	
pyyaml	5.3.1	
requests	2.24.0	
retrying	1.3.3	
scikit-image	0.17.2	
scikit-learn	0.23.2	
scipy	1.5.3	
seaborn	0.11.1	
setuptools	50.3.0	
simpleitk	2.0.1	
six	1.15.0	
sqlite	3.33.0	
tensorboard	1.13.1	
tensorflow-estimator	1.13.0	
tensorflow-gpu	1.13.1	
tensorlayer	2.2.3	
termcolor	1.1.0	
threadpoolctl	2.1.0	
tifffile	2020.10.1	
urllib3	1.25.11	
utils	1.0.1	
vc	14.1	
vs2015_runtime	14.16.27012	
werkzeug	1.0.1	
wheel	0.35.1	
wincertstore	0.2	
wrapt	1.12.1	
zipp	3.4.0	
zlib	1.2.11	
