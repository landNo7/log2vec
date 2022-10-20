import os
import csv
from tqdm import tqdm

def get_label(dir_path, filename):

	anomaly_users = []
	fin = open(os.path.join(dir_path, "anomaly_users.txt"), 'r')
	while 1:
		l = fin.readline()
		if l == '':
			break
		anomaly_users.append(l.strip("\n"))
	fin.close()

	X = []
	Y = []
	with open(os.path.join(dir_path, filename), 'r') as file:
		print("...get user label...")
		
		read = csv.reader(file)
		next(read)
		for i in tqdm(read):
			X.append(i[0])
			if i[2] in anomaly_users:
				Y.append("anomaly")
			else:
				Y.append("normal")

	return X, Y


def crop_data(dir_path, newfilename, newfilename1):
	# anomaly_users = []
	# fin = open(os.path.join(dir_path, "anomaly_users.txt"), 'r')
	# while 1:
	# 	l = fin.readline()
	# 	if l == '':
	# 		break
	# 	anomaly_users.append(l.strip("\n"))
	# fin.close()
	# logs = []
	# logs1 = []
	logs2 = []
	with open(os.path.join(dir_path, "device.csv"), 'r') as file:
		print("...get user label...")
		
		read = csv.reader(file)
		next(read)
		n = 0
		for i in tqdm(read):
			if len(logs2) < 2000:
				logs2.append(i)
			# if len(logs1) < 40000:
			# 	logs1.append(i)
			# if i[2] in anomaly_users:
			# 	logs.append(i)
			# 	continue
			# if n == 10:
			# 	logs.append(i)
			# 	n = 0
			# else:
			# 	n += 1
	# print(len(logs))
	# print(len(logs[0]))
	# with open(os.path.join(dir_path, newfilename), 'w', newline='') as file:
	# 	writer = csv.writer(file)
	# 	writer.writerow(['id', 'date', 'user', 'pc', 'activity'])
	# 	writer.writerows(logs)
	# with open(os.path.join(dir_path, newfilename1), 'w', newline='') as file:
	# 	writer = csv.writer(file)
	# 	writer.writerow(['id', 'date', 'user', 'pc', 'activity'])
	# 	writer.writerows(logs1)
	with open(os.path.join(dir_path, newfilename), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'date', 'user', 'pc', 'activity'])
		writer.writerows(logs2)

# X,Y = get_label("./data")
# print("len of X:",len(X))
# print("len of Y:",len(Y))
crop_data('./data', "crop_data_2000.csv", "crop_data1.csv")
