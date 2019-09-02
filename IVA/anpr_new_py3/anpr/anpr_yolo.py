#
# anpr_yolo
# Licene Plate detection and recognition module
# For detection it use wpod-net and
# For recognition it uses darknet
#

import sys, os
import keras
import cv2
import numpy as np
import time

from src.keras_utils 	import load_model
from glob 				import glob
from os.path 			import splitext, basename
from src.utils 			import im2single
from src.keras_utils	import load_model, detect_lp
from src.label 			import Shape, writeShapes

import darknet.python.darknet as dn
from darknet.python.darknet import detect

#from LPConsolidator import Consolidator
from anpr.LPUtils import Utils


class ANPR_YOLO():

	def __init__(self, config, det_thresh=0.5, rec_thresh=0.5):

		## --- update below versions
		self.mod_version = "1.2"
		self.det_version = "1.2"
		self.rec_version = "1.1"
		## -------------------------

		if config.det_mod == "":
			return

		#self.consolidator = Consolidator()
		self.utils = Utils()

		self.det_model = config.det_mod
		self.rec_model = config.rec_mod
		self.det_thresh = det_thresh;
		self.rec_thresh = rec_thresh;

		self.cons_lp_enabled = config.cons_lp
		self.roi_det_enabled = config.do_roi_det

		self.ocr_weights = ""
		self.ocr_netcfg = ""
		self.ocr_dataset = ""

		self.ocr_net = None
		self.ocr_meta = None

		self.wpod_net = None

		self.prev_locs = None

		if self.det_model != ""  and os.path.exists(self.det_model):
			det_model = [f for f in os.listdir(self.det_model) if ".h5" in f]
			self.det_model += "/"+det_model[0]

			print ("INFO:: Loding model: ", self.det_model)
			self.wpod_net = load_model(self.det_model)


			self.load_rec_model(self.rec_model)

		else:
			print ("\033[93m Warning: Specify detection model by giving \"--det-model <path>\" option.\033[0m")


	def version(self):
		return self.mod_version


	def print_version(self):

		if float(self.mod_version) <= 0.0:
			return -1

		print ("ANPR_MOD_VER-%s" % self.mod_version)
		print ("ANPR_DET_VER-%s" % self.det_version)
		print ("ANPR_REC_VER-%s" % self.rec_version)
		return self.version()


	def load_rec_model(self, model_path):
		
		if model_path == "" or not os.path.exists(model_path):
			print ("\033[93m Warning: Specify recognition model by giving \"--rec-model <path>\" option.'\033[0m")
			return
		
		files = os.listdir(model_path)

		for f in sorted(files):
			if self.ocr_weights == "" and ".weights" in f:
				self.ocr_weights = model_path+f
			elif self.ocr_netcfg == "" and ".cfg" in f:
				self.ocr_netcfg = model_path+f
			elif self.ocr_dataset == "" and ".data" in f:
				self.ocr_dataset = model_path+f
	
		if len(self.ocr_netcfg) > 0 and len(self.ocr_weights) > 0 and \
			len(self.ocr_dataset) > 0:
			self.ocr_net  = dn.load_net(bytes(self.ocr_netcfg, encoding='utf-8'), bytes(self.ocr_weights, encoding='utf-8'), 0)
			self.ocr_meta = dn.load_meta(bytes(self.ocr_dataset, encoding='utf-8'))


	def detect_lp_nums(self, img):
	
		if self.wpod_net is None:
			return [], -1

		h, w, c = img.shape

		ratio = float(max(img.shape[:2]))/min(img.shape[:2])
		side  = int(ratio*288.)
		bound_dim = min(side + (side%(2**4)), 608)

		lp_locs, lp_imgs, _ = detect_lp(self.wpod_net, im2single(img), bound_dim, 2**4, (240, 80), self.det_thresh)

		if len(lp_locs) == 0:
			if self.roi_det_enabled and self.prev_locs != None:
				rect, ang = self.utils.getLPCoordinates(self.prev_locs.pts, w, h)

				if self.utils.isLPGood(rect) and h-rect[1] > 80:
					w1 = int(rect[2] * 2.0);
					h1 = int(rect[3] * 4.0);
					x1 = int(rect[0] + (rect[2]/2.0) - (w1/2.0))
					y1 = int(rect[1] + (rect[3]/2.0) - (h1/2.0))

					if x1 + w1 > w:
						w1 = w - x1
					if y1 + h1 > h:
						h1 = h - y1
					
					if x1 >= 0 and y1 >=0 and w1 > 0 and h1 > 0:
						temp_img = img[y1:y1+h1, x1:x1+w1]
				
						if temp_img.shape[0] > 0 and temp_img.shape[1] > 0:

							ratio = float(max(temp_img.shape[:2]))/min(temp_img.shape[:2])
							side  = int(ratio*288.)
							bound_dim = min(side + (side%(2**4)), 608)

							lp_locs, lp_imgs, _ = detect_lp(self.wpod_net, im2single(temp_img), bound_dim, 2**4, (240, 80), self.det_thresh)

							if len(lp_locs) > 0:
								self.prev_locs = lp_locs[0]
								
								#print "Possible Detection: ", len(lp_locs)
								for lp in lp_locs:
									#r, _ = self.utils.getLPCoordinates(lp.pts, w1, h1)
									#if self.utils.isLPGood(r):
									#	cv2.rectangle(temp_img, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (0, 0, 255), 2)

									for l in range(len(lp.pts[0])):
										lp.pts[0][l] = (x1 + (lp.pts[0][l]*w1))/w
										lp.pts[1][l] = (y1 + (lp.pts[1][l]*h1))/h

									break

							else:
								self.prev_locs = None
						
							#cv2.imshow("roi", temp_img)
				else:
					self.prev_locs = None
		else:
			self.prev_locs = lp_locs[0]
				
		detected_lps = []
		lp_status = 0

		for i in range(len(lp_imgs)):

			lp_str = ""

			rect, ang = self.utils.getLPCoordinates(lp_locs[i].pts, w, h)

			if self.utils.isLPGood(rect) == False:
				continue

			if self.ocr_net is not None:
				#im_p = lp_imgs[i]
				
				im = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
				im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				im_c = tuple(np.array(im.shape[1::-1]) / 2)
				rmat = cv2.getRotationMatrix2D(im_c, ang, 1)
				im_r = cv2.warpAffine(im, rmat, im.shape[1::-1], flags=cv2.INTER_LINEAR)

				tmp_img_path = "./.temp.png"
				cv2.imwrite(tmp_img_path, im_r)
		
				res = detect(self.ocr_net, self.ocr_meta, bytes(tmp_img_path, encoding='utf-8'), thresh=self.rec_thresh)

				if len(res) > 4 and self.utils.isLPGood(rect):
					lp_str = self.utils.getLPSorted(res)			

					if self.cons_lp_enabled:
						lp_str, lp_status = self.consolidator.consolidate_lp([lp_str, res])

			detected_lps.append((lp_str, rect))

		return detected_lps, lp_status


	def consolidate(self, key):
		return self.consolidator.consolidate(key)

	def adjust_pts(pts, lroi):
		return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2, 1))


''' ----------------- Test App --------

if __name__ == '__main__':
	
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	
	showOutput = True

	wpod_net_path = 'data/lp-detector/wpod-net.h5'
	wpod_net = load_model(wpod_net_path)

	imgs_paths = sorted(glob('%s/*.jpg' % input_dir))

	print 'Searching for license plates using WPOD-NET'

	anpr = ANPR_YOLO(wpod_net_path, "data/ocr/")

	for i, img_path in enumerate(imgs_paths):

		#print '\t Processing %s' % img_path

		bname = splitext(basename(img_path))[0]
		Ivehicle = cv2.imread(img_path)

		tic = time.time()
		lp_dets = anpr.detect_lp_num(Ivehicle)
		toc = time.time()

		print "Detection + Recognition time = %.3f" % (toc-tic)
	
		if len(lp_dets) > 0:	
			for d in range(len(lp_dets)):
				print "%d, %s, %s" % (i, img_path, lp_dets[d][0])
			
				if showOutput:
					r = lp_dets[d][1]
					cv2.rectangle(Ivehicle, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 2)
					cv2.putText(Ivehicle, lp_dets[d][0], (r[0], r[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
				print "%d, %s, " % (i, img_path)
			

		if showOutput:
			cv2.imshow("ANPR", Ivehicle)
			k = cv2.waitKey(10)

			if k == 27:
				break

		if output_dir:
			cv2.imwrite("%s/frame_%04d.jpg" % (output_dir, i), Ivehicle);
''' 


