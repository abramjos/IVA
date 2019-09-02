
class Config():

	def __init__(self):
		self.config = {}

		self.cam_url    = "" 
		self.out_path   = ""
		self.det_mod    = ""
		self.rec_mod    = ""
		self.silence    = True
		self.show_out   = False
		self.test_mode  = False
		self.show_fps   = False
		self.skip       = 0
		self.cons_lp    = False
		self.do_roi_det = False


	def read(self, config_file):
		fp = open(config_file)
		while True:
			line = fp.readline()
			if not line:
				break

			if len(line) < 3 or line[0] == '#':
				continue;

			key, val = line.strip().split('=')
			#print "======", key, val
			self.config[key] = val
		
		fp.close()

		self.load(self.config)

		#self.cam_url   = self.config['INPUT']
		#self.out_path  = self.config['OUTPUT']
		#self.det_mod   = self.config['DET_MODEL']
		#self.rec_mod   = self.config['REC_MODEL']
		#self.skip      = int(self.config['SKIP_FRAME'])
		#if self.config['SHOW'] == "TRUE": self.show_out = True
		#if self.config['OUT_RES'] == "TRUE": self.silence = False
		#if self.config['TEST_MODE'] == "TRUE": self.test_mode = True
		#if self.config['OUT_FPS'] == "TRUE": self.show_fps = True
		#if self.config['CONS_LP'] == "TRUE": self.cons_lp = True
		#if self.config['DET_ROI'] == "TRUE": self.do_roi_det = True

		return self.config


	def load(self, config_dict):
		self.cam_url   = config_dict['INPUT']
		self.out_path  = config_dict['OUTPUT']
		self.det_mod   = config_dict['DET_MODEL']
		self.rec_mod   = config_dict['REC_MODEL']
		self.skip      = int(config_dict['SKIP_FRAME'])
		if config_dict['SHOW'] == "TRUE": self.show_out = True
		if config_dict['OUT_RES'] == "TRUE": self.silence = False
		if config_dict['TEST_MODE'] == "TRUE": self.test_mode = True
		if config_dict['OUT_FPS'] == "TRUE": self.show_fps = True
		if config_dict['CONS_LP'] == "TRUE": self.cons_lp = True
		if config_dict['DET_ROI'] == "TRUE": self.do_roi_det = True

		#self.cam_url   = config_dict['INPUT']
		#self.out_path  = config_dict['OUTPUT']
		#self.det_mod   = config_dict['DET_MODEL']
		#self.rec_mod   = config_dict['REC_MODEL']
		#self.show_out  = config_dict['SHOW']
		#self.silence   = config_dict['OUT_RES']
		#self.skip      = config_dict['SKIP_FRAME']
		#self.test_mode = config_dict['TEST_MODE']
		#self.show_fps  = config_dict['OUT_FPS']


	def show(self):
		for key in self.config:
			print (key, ":", self.config[key])
