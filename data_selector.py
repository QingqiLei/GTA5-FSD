import sys
import random
import csv
from collections import Counter
import numpy as np
import os
import utils

label_csv = utils.label_file

	
entries=[]
right_samples=0
left_samples=0
straight_samples=0

not_thr1 = 0
not_brake = 0

with open(label_csv, 'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	i = 0
	for row in csv_reader:
		i+=1
		if i == 1:
			continue
		
		steering_angle=float(row[2])
		
		if (steering_angle > 0.5):
			right_samples+=1
		elif (steering_angle < 0.5):
			left_samples+=1
		else:
			straight_samples+=1
			
		throttle=float(row[3])
		brake=float(row[4])
		if throttle == 1:
			not_thr1 +=1
		if brake == 1:
			not_brake +=1

		entries.append((steering_angle, throttle, brake))

counter = Counter(tuple(tup) for tup in entries)

most_common=counter.most_common(20)

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


	