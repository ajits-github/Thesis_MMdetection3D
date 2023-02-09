![ttc_crop](https://user-images.githubusercontent.com/71327507/217001177-89c35fd3-16b2-4900-b0bb-4658262b2a85.png)
---------------------------------------------------------------------------------------------------------------------------------


## Time-to-Collision Estimation for Objects in Autonomous Driving

* The research aims to calculate the time for the ego vehicle to collide with the objects in the view
* The dataset has been prepared using nuScenes and model used is FCOS3D (Fully Convolutional One-Stage Monocular 3D Object Detection) from MMDetection3D framework.
* FCOS3D is a general anchor-free, one-stage monocular 3D object detector adapted from the original 2D version FCOS. It serves as a baseline built on top of mmdetection and mmdetection3d for 3D detection based on monocular vision.
* MMDetection3D is an open source object detection toolbox based on PyTorch, towards the next-generation platform for general 3D detection. It is a part of the OpenMMLab project developed by MMLab.
* The architecture of the model has been altered by adding the convolution and relu layers for ttc (time to collision)
* The loss function is implemented from the paper - "Binary TTC: A Temporal Geofence for Autonomous Navigation" called Motion in Depth(MiD) error
* Architecture of the network includes- ResNet101 as backbone, Feature Pyramid Network(FPN) as neck and FCOSMono3D as head
* Below is the nuScenes dataset meta information  
---------------------------------------------------------------------------------------------------------------------------------
#### The ego vehilce used for preparation of nuScenes dataset along with all the sensors.
![car](https://user-images.githubusercontent.com/71327507/216977031-424a117f-1634-4cb1-a316-534c3da525cb.jpeg)

---------------------------------------------------------------------------------------------------------------------------------
#### An image sample with all the 3D bboxes for a sample data token i.e. images taken from all the sensors at the same time.
![Images_all_camera_bboxes](https://user-images.githubusercontent.com/71327507/216998094-ef3635e6-aa7d-4d69-a221-d76db56f9422.png)

---------------------------------------------------------------------------------------------------------------------------------
#### The dataset distribution
![nuscenes_distribution1](https://user-images.githubusercontent.com/71327507/216977188-3bf35045-19b2-4166-a8b1-f63a9917a56c.png)

