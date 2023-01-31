## Time-to-Collision Estimation for Objects in Autonomous Driving

* The research aims to calculate the time for the ego vehicle to collide with the objects in the view
* The dataset has been prepared using nuScenes and model used is FCOS3D (Fully Convolutional One-Stage Monocular 3D Object Detection) from MMDetection3D framework.
* FCOS3D is a general anchor-free, one-stage monocular 3D object detector adapted from the original 2D version FCOS. It serves as a baseline built on top of mmdetection and mmdetection3d for 3D detection based on monocular vision.
* MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is a part of the OpenMMLab project developed by MMLab.
* The architecture of the model has been altered by adding the convolution and relu layers for ttc (time to collision)
* The loss function is implemented from the paper - "Binary TTC: A Temporal Geofence for Autonomous Navigation" called Motion in Depth(MiD) error
* Architecture of the network includes- ResNet101 as backbone, Feature Pyramid Network(FPN) as neck and FCOSMono3D as head
