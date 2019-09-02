
import numpy as np

import math

class Utils():


	def LD(self, seq1, seq2):
		size_x = len(seq1) + 1
		size_y = len(seq2) + 1
		matrix = np.zeros ((size_x, size_y))

		for x in xrange(size_x):
			matrix [x, 0] = x
		for y in xrange(size_y):
			matrix [0, y] = y

		for x in xrange(1, size_x):
			for y in xrange(1, size_y):
				if seq1[x-1] == seq2[y-1]:
					matrix [x,y] = min(
						matrix[x-1, y] + 1,
						matrix[x-1, y-1],
						matrix[x, y-1] + 1)
				else:
					matrix [x,y] = min(
						matrix[x-1,y] + 1,
						matrix[x-1,y-1] + 1,
						matrix[x,y-1] + 1)

		#print (matrix)
		return (matrix[size_x - 1, size_y - 1])


	def isLPDoubleRow(self, res):
		sz = len(res)
		double_row_idx = -1

		for r in range(1, sz):
			y_diff = abs(res[r-1][2][1] - res[r][2][1])
			char_height = res[r][2][3]

			#FIXME: -------------------------   check below condition
			if y_diff > (char_height/2.0) and y_diff < (char_height*2.0):
				double_row_idx = r
				break

		return double_row_idx


	def getLPSorted(self, res):
		# ------ Sort LP based on position -------
		res.sort(key=lambda x: x[2][1])

		double_row = self.isLPDoubleRow(res)

		if double_row >= 0:
			sorted_res = sorted(res[:double_row], key=lambda x: x[2][0])
			sorted_res += sorted(res[double_row:], key=lambda x: x[2][0])
			res = sorted_res
		else:
			res.sort(key=lambda x: x[2][0])

		lp_str = b"".join([r[0] for r in res])

		return lp_str.decode('utf-8')


	def getLPCoordinates(self, pts, w1, h1, pad=5):
		#print pts
		top = pts[0].tolist()
		bot = pts[1].tolist()

		x1, y1 = int(top[0] * w1), int(bot[0] * h1)
		x2, y2 = int(top[1] * w1), int(bot[1] * h1)

		del_x, del_y = x2-x1, y2-y1
		
		rad = math.atan2(del_y, del_x)
		ang1 = math.degrees(rad)

		x3, y3 = int(top[3] * w1), int(bot[3] * h1)
		x4, y4 = int(top[2] * w1), int(bot[2] * h1)
		
		del_x, del_y = x4-x3, y4-y3
		
		rad = math.atan2(del_y, del_x)
		ang2 = math.degrees(rad)

		ang = (ang1+ang2)/2.0	

		x = x1 - pad
		y = y1 - pad
		w = int((top[1] - top[3]) * w1) + 2*pad
		h = int((bot[2] - bot[0]) * h1) + 2*pad

		return (x, y, w, h), ang


	def isLPGood(self, lp_box):
		if lp_box[0] < 0 or lp_box[1] < 0:
			return False

		if lp_box[2] < 30 or lp_box[2] > 150 or lp_box[3] < 10 or lp_box[3] > 100:
			return False
		
		return True
