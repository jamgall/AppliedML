import csv
import random

def main():
	to_write_train = []
	to_write_test = []
	with open('data.csv', 'rb') as file:
		csvfile = csv.reader(file)
		for row in csvfile:
			#print row
			x = random.randint(1,11)
			if(x<8):
				to_write_train.append(row)
			else:
				to_write_test.append(row)

	with open('train.csv', 'w') as file:
		csvwriter = csv.writer(file)
		for row in to_write_train:
			if row[0] == 'B':
				row[0] = 0
			else:
				row[0] = 1
			csvwriter.writerow(row)


	with open('test.csv', 'w') as file:
		csvwriter = csv.writer(file)
		for row in to_write_test:
			if row[0] == 'B':
				row[0] = 0
			else:
				row[0] = 1
			csvwriter.writerow(row)


if __name__ == '__main__':
	main()