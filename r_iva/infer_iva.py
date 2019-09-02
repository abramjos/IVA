import sys
sys.path.insert(0, './anpr_new_py3');


import random
import anpr_wrap as anpr
from trt_engine_classifier import engine

import json
from load_ssd  import load_ssd
import cv2
import numpy as np
import json
import time
import argparse
import colorsys


class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, (complex, np.complex)):
            return [obj.real, obj.imag]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, bytes):  
            return obj.decode()
        return json.JSONEncoder.default(self, obj)

class iva_infer():
    def __init__(self,path='./'):
        self.detector=load_ssd()

        fold=[path+'Secondary_CarColor',path+'Secondary_CarMake',path+'Secondary_VehicleTypes']
        self.carmake=engine(fold[1]+'/resnet18.caffemodel_b16_int8.cache', fold[1]+'/labels.txt')
        self.carcolor=engine(fold[0]+'/resnet18.caffemodel_b16_int8.cache', fold[0]+'/labels.txt')
        self.cartype=engine(fold[2]+'/resnet18.caffemodel_b16_int8.cache', fold[2]+'/labels.txt')

        self.anpr= anpr
        self.anpr.load_model()

        hsv_tuples = [(x / len(self.detector.classes), 1., 1.)
              for x in range(len(self.detector.classes))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None) 

    def infer(self,image,image_id=0):
        t1=time.time()
        val=self.detector.predict(image,image_id)
        if val['success']==True:
            for _id,key in enumerate(val['results']['detections']):
                im=key['im']
                image=cv2.resize(im,(224,224))                   
                car_color=self.carcolor.predict(image)
                car_make=self.carmake.predict(image)
                car_type=self.cartype.predict(image)
                anpr_out=self.anpr.detect(im)

                val['results']['detections'][_id]['car-color']=car_color
                val['results']['detections'][_id]['car-type']=car_type
                val['results']['detections'][_id]['car-make']=car_make

                if anpr_out['success']==True:
                    val['results']['detections'][_id]['anpr']=anpr_out['anpr'][0][0]
                    _box=val['results']['detections'][_id]['boundingBox']
                    anpr_box=anpr_out['anpr'][0][1]
                    val['results']['detections'][_id]['anpr_box']=[_box['top']+anpr_box[1],_box['left']+anpr_box[0],anpr_box[3],anpr_box[2]]
                else:
                    val['results']['detections'][_id]['anpr']=None
                val['results']['detections'][_id].pop('im',None)
        val['results']['timeUsed']=time.time()-t1
        return(val)

    def _im_panel(self,result,im):
        thickness = (im.shape[0] + im.shape[1]) // 200
        if result['success']==True:
            for detection in result['results']['detections']:
                _class=detection['class']
                _box=detection['boundingBox']
                _color=self.colors[detection['objectId']]
                _car_color=detection['car-color']
                _car_type=detection['car-type']
                if detection['anpr']!=None:
                    _anpr=detection['anpr']
                    _anpr_box=detection['anpr_box']
                    cv2.rectangle(im,(_anpr_box[1],_anpr_box[0]),(_anpr_box[1]+_anpr_box[3],_anpr_box[0]+_anpr_box[2]),self.colors[-2],thickness=thickness)
                    cv2.putText(im, _anpr,(_anpr_box[1]+3, _anpr_box[0]-3), cv2.FONT_HERSHEY_SIMPLEX, .5, self.colors[-2], thickness=thickness)

                cv2.rectangle(im,(_box['top'],_box['left']),(_box['top']+_box['width'],_box['left']+_box['height']),_color,thickness=thickness)
                cv2.putText(im, '{}-{}-{}'.format(_class,_car_color,_car_type),(_box['top']+3, _box['left']+15), cv2.FONT_HERSHEY_SIMPLEX, .7, _color, thickness=thickness)
        return(im)

    def infer_video(self,vid_path):
    
        vid = cv2.VideoCapture(vid_path)
        # vid.set(cv2.CAP_PROP_FRAME_WIDTH,self.detector.im_w)
        # vid.set(cv2.CAP_PROP_FRAME_HEIGHT,self.detector.im_h)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps=vid.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, vid_fps, (width,height))
        count=0

        while (vid.isOpened()):
            return_value, frame = vid.read()
            if return_value==True and count < 2000:
                result=self.infer(frame,image_id=count)
                frame=self._im_panel(result,frame)
                print(result)
                out.write(frame)
                cv2.imshow("Inference Output",frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break


if __name__ == '__main__':




    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--path',type=str,default='sample.jpg',
                        help='image or video path')
    parser.add_argument('-v','--vid_flag',type=bool,default=False)
    parser.add_argument('-o','--json_path',type=str,default='data.json',
                        help='processed output json for images')
    args = parser.parse_args()
    
    inference_engine=iva_infer()

    if args.vid_flag==False:    
        image=cv2.imread(args.path)
        val=inference_engine.infer(image) 
        json_val=json.dumps(val,cls=JsonCustomEncoder)
        with open(args.json_path, 'w') as outfile:
            json.dump(json_val, outfile)
        imx=inference_engine._im_panel(val,image)
        cv2.imwrite('x.jpg',imx)
#        import ipdb;ipdb.set_trace()
    else:
        sample_video='/harddisk/abraham/video/7_20190114_203847.avi'
        inference_engine.infer_video(sample_video)

