import csv

def main():
	stances = {}
	with open('train_bodies.csv', 'rb') as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			stances[row[0]] = row[1]

	to_write = []
	with open('train_stances.csv', 'rb') as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			to_write.append([row[2],row[0],stances[row[1]]])


	with open('train.csv', 'w') as file:
		csvwriter = csv.writer(file)
		for row in to_write:
			csvwriter.writerow(row)

	stancests = {}
	with open('competition_test_bodies.csv', 'rb') as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			stancests[row[0]] = row[1]

	to_writets = []
	with open('competition_test_stances.csv', 'rb') as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			to_writets.append([row[2], row[0], stancests[row[1]]])

	with open('test.csv', 'w') as file:
		csvwriter = csv.writer(file)
		for row in to_writets:
			csvwriter.writerow(row)

if __name__ == '__main__':
	main()