import pickle

# with open('C:\\Users\\ajit1\\Google Drive\\Colab Notebooks\\Thesis\\converted\\kitti_infos_train.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data)


import cv2
img = cv2.rectangle("C:\\Users\\ajit1\\Google Drive\\Colab Notebooks\\Thesis\\v1.0-mini\\samples\\CAM_FRONT\\n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg", (1206.5693751819115, 477.86111828160216), (19.31993062031279, 35.78389940122628), (255,0,0), 2)
cv2.imwrite("my.png",img)