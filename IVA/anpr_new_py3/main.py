import anpr_wrap as anpr

import sys, os
import cv2

def load_files(path):
	files = [os.path.join(path, f) for f in os.listdir(path)]
	return files

def main():
	files = load_files(sys.argv[1])
	anpr.load_model()
	
	for f in files:
            print (f)
            img = cv2.imread(f)
            ret = anpr.detect(img)
            print (ret)

if __name__ == "__main__":
	main()
