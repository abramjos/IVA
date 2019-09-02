# 
# anpr.py
# Wrapper over anpr_yolo and to deal with 
# camera and other stuffs
# 

import os, sys, time

import cv2, pdb

from anpr.anpr_yolo import ANPR_YOLO
from anpr.LPUtils import Utils
from anpr.config import Config


anpr_g = None


def load_config(argv):

	config = Config()

	conf = {}
	
	if os.path.isfile("./anpr_new_py3/anpr.cfg"):
		conf = config.read("./anpr_new_py3/anpr.cfg")

	for i in range(1, len(argv), 2):
		if argv[i] == '-i' or argv[i] == '--input': 
			conf['INPUT'] = argv[i+1]
		elif argv[i] == '-o' or argv[i] == '--output': 
			conf['OUTPUT'] = argv[i+1]
		elif argv[i] == '-c' or argv[i] == '--config': 
			conf = config.read(config_path)
		elif argv[i] == '-dm' or argv[i] == '--det-model':
			conf['DET_MODEL'] = argv[i+1]
		elif argv[i] == '-rm' or argv[i] == '--rec-model':
			conf['REC_MODEL'] = argv[i+1]
		elif argv[i] == '-sh' or argv[i] == '--show':
			conf['SHOW'] = int(argv[i+1]) == 1
		elif argv[i] == '-sf' or argv[i] == '--fps':
			conf['OUT_FPS'] = int(argv[i+1]) == 1
		elif argv[i] == '-s' or argv[i] == '--silence':
			conf['OUT_RES'] = int(argv[i+1]) == 1
		elif argv[i] == '-sk' or argv[i] == '--skip':
			conf['SKIP_FRAME'] = int(argv[i+1])
		elif argv[i] == '-t' or argv[i] == '--test':
			conf['TEST_MODE'] = 1
		elif argv[i] == '-cl' or argv[i] == '--cons':
			conf['CONS_LP'] = 'TRUE'
		elif argv[i] == '-dr' or argv[i] == '--droi':
			conf['DET_ROI'] = 'TRUE'
		elif argv[i] == '-v' or argv[i] == '--version':
			print (print_version())
			return 0
		elif argv[i] == '-h' or argv[i] == '--help':
			print (print_usage(argv))
			return 0
		else:
			print ("Not an option: %s" % argv[i])
			return -1

	config.load(conf)

	return config

def load_model(config=[]):
    global anpr
    config = load_config(config)
    anpr = ANPR_YOLO(config)

def detect(img):
    global anpr
    if anpr is not None:
        lp_dets, lp_status = anpr.detect_lp_nums(img)
        flag = False
        for ix,lp in enumerate(lp_dets):
            if len(lp[0]) > 0:
                flag = True
        return {"anpr": lp_dets, "success": flag}
    else:
        return {"anpr": "%s" % lp_dets, "success": True}


def print_usage(argv):
	print ("python %s --input <video-path/cam-url def=''>" % argv[0]) 
	print ("               --output <video-path        def=''>")
	print ("               --det-model <path-det-mod   def='models/lp_model_1.2/'>")
	print ("               --rec-model <path-rec-mod   def='models/ocr_model_1.1/'>") 
	print ("               --show <show-output         def='False'>")
	print ("               --silence <discard-couts    def='True'>")
	print ("               --skip <skip-frames         def='0'>")
	print ("               - Press <esc*2> to exit.")


def print_version():
	anpr = ANPR_YOLO()
	vers = anpr.print_version()


def read_file(fpath):
	files_list_g = []

	with open(fpath, 'r') as fp:
		while True:
			 line = fp.readline().strip()
			 if not line:
				 break

			 files_list_g.append(line)

	return 1, files_list_g


def read_dir(dpath):
	files_list_g = [dpath+f for f in os.listdir(dpath) if ".jpg" in f]
	
	return 2, files_list_g


def read_video(url):
	ret, cam = read_camera(url)
	return 3, cam
	

def read_camera(url):
	cap = cv2.VideoCapture(url)
	if cap.isOpened() == False:
		print ("Can't open: %s" % url)
		return -1, -1

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
 
	return 4, cap
	

def init_source(url):

	status = -1
	module = None

	if os.path.isfile(url): 
		if url[:-3] == "txt":
			status, module = read_file(url);
		else:
			status, module = read_video(url)
	elif os.path.isdir(url):
		status, module = read_dir(url)
	else:
		status, module = read_camera(url)

	return status, module
		

def get_new_frame(source_id, source, fcount):
   
	if source_id == 1 or source_id == 2:
		if fcount < len(source):
			return source_id, cv2.imread(source[fcount])
		else:
			return -1, -1
	elif source_id == 3 or source_id == 4:
		return source.read()


#def main():
#	
#	config = load_config(sys.argv)
#
#	if config.cam_url == "":
#		print (print_usage(sys.argv))
#		return -1
#	else:
#		print ("Source:", config.cam_url)
#
#	config.show()
#
#	idx, source = init_source(config.cam_url)
# 
#	if len(config.out_path) > 0:
#		if not os.path.exists(config.out_path):
#			os.makedirs(config.out_path)
#
#		filename = os.path.basename(config.cam_url)
#		output_video_file = "%s/processed_%s" %(config.out_path, filename[:-4]+".avi")
#		print ("Output video name: ", output_video_file)
#		fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#		out_vid = cv2.VideoWriter(output_video_file, fourcc, int(12), (int(source.get(3)), int(source.get(4))))
#
#	utils = Utils()
#
#	anpr = ANPR_YOLO(config)
#	vers = anpr.print_version()
#	detn, break1, fc = 0, 0, 0
#
#	start_time = time.time()
#	fps, no_detection = 0, 0
#	lp_str, prev_lp = "", ""
#
#	last_frame = None
#	detected = 0
#
#	while (break1 < 2):
#		# Capture frame-by-frame
#		ret, frame = get_new_frame(idx, source, fc)
#		if ret <= 0:
#			break
#
#		if (fc < config.skip and idx != 4) or (idx == 4 and config.skip <= 5 and fc%config.skip == 0):
#			fc += 1
#			fps += 1
#			continue
#
#		tic = time.time()
#		lp_dets, lp_status = anpr.detect_lp_nums(frame)
#		toc = time.time()
#
#		dets = len(lp_dets)
#
#		if config.show_fps:
#			run_time = toc - start_time
#			if run_time > 5.0:
#				print ("INFO:: FPS = %.2f" % (fps/run_time))
#				start_time = toc
#				fps = 0
# 
#		if dets > 0:
#			detn += 1
#			detected = 0
#			for d in range(dets):
#					
#				if config.silence == False:
#					print (fc, "# Detected: %s, time=%.3f sec(s)" % (lp_dets[d][0], toc-tic))
#
#				r = lp_dets[d][1]
#				if len(lp_dets[d][0]) > 4 and len(lp_dets[d][0]) < 14 and (config.show_out or len(config.out_path) > 0):
#					prev_lp = lp_str
#					lp_str = lp_dets[d][0]
#						
#					cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 255, 0), 2)
#					cv2.putText(frame, lp_str, (r[0], r[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#					# save even frames -------- for testing only
#					if False and lp_status == 1:
#						out_name = "output/frame_%06d_%s.jpg" % (fc, prev_lp)
#						print ("==>", no_detection, "(", lp_str, prev_lp, ")")
#						print ("Saving output:", out_name)
#		
#						cv2.imwrite(out_name, last_frame)
#						no_detection = 0
#						#anpr.consolidate(prev_lp) # clear stacked up data
#
#					last_frame = frame
#					# -------------------------------------------
#					detected = 1
#				elif config.show_out and utils.isLPGood(r):
#					cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, 0, 255), 2)
#					no_detection += 1
#					detected = 1
#				else:
#					cv2.rectangle(frame, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255, 0, 0), 2)
#	
#		fc += 1
#		fps += 1
#
#		if config.show_out:
#			cv2.imshow("ANPR Running..", frame)
#			k = cv2.waitKey(10)
#
#			if k == 27:
#				break1 += 1
#
#		if len(config.out_path) > 0:
#			out_vid.write(frame)
#
#	if config.test_mode:
#		print ("Detection Accuracy = %.2f" % (detn/float(fc)))
#
#	if len(config.out_path) > 0:
#		out_vid.release()
#		
#	cv2.destroyAllWindows()
#
#
#if __name__=="__main__":
#	main()
#
