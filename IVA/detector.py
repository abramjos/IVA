from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import cv2
import sys
sys.path.insert(0, './ssd_keras')
from models.keras_ssd300 import ssd_300


class detector():
    def __init__(self,required_class=[2,6,7,14,15],weights_path='./VGG_VOC0712Plus_SSD_300x300_ft_iter_160000.h5',img_height = 300,img_width = 300):


        self.img_height,self.img_width=img_height,img_width

        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=20,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                 [1.0, 2.0, 0.5],
                                                 [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        self.classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
        self.required_class=required_class

        self.model.load_weights(weights_path, by_name=True)



    def predict(self,image,confidence_threshold = 0.5):

        image_resize=cv2.resize(image,(self.img_width,self.img_height))
        image_reshape=image_resize.reshape((1,self.img_width,self.img_height,3))
        y_pred = self.model.predict(image_reshape)


        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_thresh[0])
        result=[]

        for box in y_pred_thresh[0]:
            if  int(box[0]) in self.required_class:       
                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                xmin = max(0,int(box[2] * (image.shape[1] / float(self.img_width))))
                ymin = max(0,int(box[3] * (image.shape[0] / float(self.img_height))))
                xmax = min(image.shape[1],int(box[4] * (image.shape[1] / float(self.img_width))))
                ymax = min(image.shape[0],int(box[5] * (image.shape[0] / float(self.img_height))))
                
                label,conf = self.classes[int(box[0])], box[1]
                obj=image[ymin:ymax,xmin:xmax,:]

                result.append([label,conf,obj,xmin,xmax,ymin,ymax])
        # result=np.array(result)
        return(result)
if __name__ == '__main__':
        
    x=detector()
    im=cv2.imread('/home/graymatics/Deep/DeepStream_Release/samples/carshape/keras/test_fold/buick-lucerne-frontside_buluc111.jpg')
    s=x.predict(im)


