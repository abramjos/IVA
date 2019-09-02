
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob 

from random import randint
from PIL import Image
# from matplotlib.pyplot import imshow #to show test case

from tensorrt import parsers



class engine():
	def __init__(self, engine_file, labels,classifier=True):

		self.classifier=classifier
		if self.classifier:
			file = open(labels, 'r')
			lst=file.read()
		else:
			print('sorry')

		self.LABELS=lst.replace('\n','').split(';')
		self.output = np.empty(len(self.LABELS), dtype = np.float32)


		self.G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
		self.engine = trt.utils.load_engine(self.G_LOGGER,engine_file)

		self.runtime = trt.infer.create_infer_runtime(self.G_LOGGER)
		self.context = self.engine.create_execution_context()



	def predict(self, image):
		assert(self.engine.get_nb_bindings() == 2)
		#convert input data to Float32
		image = image.astype(np.float32)
		d_input = cuda.mem_alloc(1 * image.size * image.dtype.itemsize)
		d_output = cuda.mem_alloc(1 * self.output.size * self.output.dtype.itemsize)
		bindings = [int(d_input), int(d_output)]

		stream = cuda.Stream()		
		cuda.memcpy_htod_async(d_input, image, stream)
		self.context.enqueue(1, bindings, stream.handle, None)
		cuda.memcpy_dtoh_async(self.output, d_output, stream)
		stream.synchronize()

		prediction=self.LABELS[np.argmax(self.output,axis=0)]

		return(prediction)

	def close_engine(self):
		print("Deleting engines..")
		self.context.destroy()
		self.engine.destroy()
		self.runtime.destroy()


if __name__ == '__main__':

	fold=['Secondary_CarColor','Secondary_CarMake','Secondary_VehicleTypes']
	car_make=engine(fold[1]+'/resnet18.caffemodel_b16_int8.cache', fold[1]+'/labels.txt')

	car_color=engine(fold[0]+'/resnet18.caffemodel_b16_int8.cache', fold[0]+'/labels.txt')

	car_type=engine(fold[2]+'/resnet18.caffemodel_b16_int8.cache', fold[2]+'/labels.txt')
			
	im_list=glob.glob('/home/graymatics/Deep/DeepStream_Release/samples/carshape/keras/test_fold/bmw*')

	for im_path in im_list:
		img=cv2.imread(im_path)
		print(im_path)
		print(car_make.predict(img))
		print(car_color.predict(img))
		print(car_type.predict(img))





