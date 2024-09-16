import sys
import random
import csv
from collections import Counter
import numpy as np
import os
import utils
import random




def analyze(file_name):
	entries=[]
	right_samples=0
	left_samples=0
	straight_samples=0

	not_thr1 = 0
	not_brake = 0

	t1 = 0
	t2 = 0

	with open(file_name, 'r') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		i = 0
		for row in csv_reader:
			i+=1
			if i == 1:
				continue
			
			steering_angle=float(row[2])
			
			if (steering_angle > 0.5):
				right_samples+=1
				t2 += (steering_angle - 0.5)
			elif (steering_angle < 0.5):
				left_samples+=1
				t1 += (0.5 - steering_angle)
			else:
				straight_samples+=1
				
			throttle=float(row[3])
			brake=float(row[4])
			if throttle == 0:
				not_thr1 +=1
			if brake == 0:
				not_brake +=1

			entries.append((steering_angle, throttle, brake))

	counter = Counter(tuple(tup) for tup in entries)

	most_common=counter.most_common(20)

	print(t1, t2)

	print("Total Samples: %d\n" % len(entries))
	print("Average counts: %.3f" % np.mean(list(counter.values())))
	print("Average counts (most common): %.3f\n" % np.mean([count for key, count in most_common]))

	print("Right steer samples: %d (%.3f%% of total samples)" % (right_samples, (right_samples/len(entries))*100))
	print("Left steer samples: %d (%.3f%% of total samples)" % (left_samples, (left_samples/len(entries))*100))
	print("Straight steer samples: %d (%.3f%% of total samples)\n" % (straight_samples, (straight_samples/len(entries))*100))
	print("Not throttle samples: %d (%.3f%% of total samples)" % (not_thr1, (not_thr1/len(entries))*100))
	print("Not brake samples: %d (%.3f%% of total samples)\n" % (not_brake, (not_brake/len(entries))*100))

	for index in range(len(most_common)):
		elem = most_common[index]
		print("%d.\t%.3f%%  [%.6f  %.6f  %.6f] -> %d" % (index+1, 100*(elem[1]/len(entries)), elem[0][0], elem[0][1], elem[0][2], elem[1]))


def create_new_label_file(old_file, new_file_path):

	lines = open(old_file, 'r').readlines()
	
	with open(new_file_path, 'w') as output_file:
		output_file.write(lines[0])
		for line in lines[1:]:
			file_name, speed, steer, throttle, brake = line.split(',')
			file_name, speed, steer, throttle, brake = file_name, float(speed), float(steer), float(throttle), float(brake.strip())
			
			if (steer == 0.5 and throttle == 1 and brake == 1):
				if random.random() > 0.2:
					continue
			if (steer > 0.5):
				if random.random() > 0.95:
					continue

			output_file.write('%s,%f,%f,%f,%f\n' % (file_name, speed, steer, 1- throttle, 1- brake))

		output_file.flush()

	print('new file length:', len(open(new_file_path, 'r').readlines()))
		

if __name__ == '__main__':
	label_csv = utils.label_file
	analyze(label_csv)
	new_file_path = os.path.join(utils.data_dir, 'tmp.csv')
	create_new_label_file(label_csv, new_file_path)
	analyze(new_file_path)
