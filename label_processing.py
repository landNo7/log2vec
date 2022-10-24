import os
import csv
from tqdm import tqdm


def get_label(dir_path, filename, labelfile):

	anomaly_users = []
	fin = open(os.path.join(dir_path, labelfile), 'r')
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
		num = 0
		for i in tqdm(read):
			num += 1
			if i[0] == "":
				i[0] = X[-1]
			X.append(i[0])
			if num < 1000 and i[0] in anomaly_users:
				Y.append("anomaly")
			else:
				Y.append("normal")

	return X, Y


def crop_data(dir_path, newfilename, newfilename1):
	logs2 = []
	with open(os.path.join(dir_path, "device.csv"), 'r') as file:
		print("...get user label...")
		
		read = csv.reader(file)
		next(read)
		n = 0
		for i in tqdm(read):
			if len(logs2) < 2000:
				logs2.append(i)
	with open(os.path.join(dir_path, newfilename), 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'date', 'user', 'pc', 'activity'])
		writer.writerows(logs2)

# crop_data('./data', "crop_data_2000.csv", "crop_data1.csv")
# X,Y = get_label('./data', "attack.csv", "anomaly_list.txt")
# for i,j in zip(X,Y):
# 	print(i,j)
# print(len(X))