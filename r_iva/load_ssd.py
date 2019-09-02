import system
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time

class load_ssd():
  def __init__(self,pb_file="./tf_ssd.pb",required_class=[2,6,7,14],conf=.5, img_height = 300,img_width = 300):
    self.sess=tf.Session()
    if (os.path.isfile('./tf_ssd.pb') == False):
        print("Downloading tensorflow ssd weights for detection")
        os.system('wget -O ./tf_ssd.pb https://www.dropbox.com/s/reg9e9sa1thovgt/tf_ssd.pb?dl=0')
    self.im_h=300
    self.im_w=300
    self.conf=conf
    self.sess.graph.as_default()
    self.classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
    self.required_class=required_class

    f = gfile.FastGFile(pb_file, 'rb')
    graph_def = tf.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()
    self.sess.graph.as_default()
    tf.import_graph_def(graph_def)
    self.softmax_tensor = self.sess.graph.get_tensor_by_name('import/decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3:0')
    return(None)

  def predict(self,image,image_id):

    im_resize=cv2.resize(image,(self.im_w,self.im_h))
    im_tensor=im_resize.reshape((1,self.im_w,self.im_h,3))

    predictions = self.sess.run(self.softmax_tensor, {'import/input_1:0': im_tensor})[0]
    
    predict=self._predict(predictions,image,image_id)
    return(predict)

  def _predict(self,predictions,image,image_id):
    t1=time.time()
    im_h,im_w=image.shape[:2]
    predictions=predictions[predictions.nonzero()]
    predict=predictions.reshape(int(len(predictions)/6),6)
    out={}
    result={}

    try:
      result['imageId']=image_id
      result['requestId']='0'
      detections=[]

      _id=0
      try:  
        for i in predict:
          _id+=1
          if i[1]>self.conf and int(i[0]) in self.required_class:
            obj={}
            obj['class']=self.classes[int(i[0])]
            obj['objectId']=int(i[0])
            xmin = max(0,int(i[2] * (im_w / float(self.im_w))))
            ymin = max(0,int(i[3] * (im_h / float(self.im_h))))
            xmax = min(im_w,int(i[4] * (im_w / float(self.im_w))))
            ymax = min(im_h,int(i[5] * (im_h / float(self.im_h))))
            obj['boundingBox']={ "top":  xmin, "left":  ymin, "width":   xmax-xmin, "height":  ymax-ymin }

            obj['confidence']=i[1]
            obj['im']=image[ymin:ymax,xmin:xmax,:]
            detections.append(obj)
        out['success']=True
      except:      
        out['success']=False
      result['detections']=detections
      result['timeUsed']=time.time()-t1
      out['results']=result

    except:
      out['success']=False
      out['results']=result

    return(out)



if __name__ == '__main__':
  x=cv2.imread('sample.jpg')
  print(x.shape)
  s=load_ssd()
  print(s.predict(x))

