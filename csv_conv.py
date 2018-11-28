import pandas as pd
import numpy as np
import sys,os
import csv
csv.field_size_limit(sys.maxsize)
from vecshare import vecshare as vs

def converter(filename):
	shortName = filename.split('/')[-1]
	with open(filename, 'r') as fi:
		cnt = 0
		while True:
			evan = []
			block = fi.readlines(1024*1024*512)
			if not block:
				break
			for line in block:
				line = line.strip('\n').split(' ')
				evan.append(line)
			df= pd.DataFrame(evan)
			df.to_csv('%s_%d.csv' % (shortName, cnt), index = False, header = False, quoting=csv.QUOTE_NONE, escapechar= '\\' )
			cnt+=1
			del evan[:], df
	# combine them together using unix
	os.system("cat %s* > output.csv"%(shortName))
	os.system("rm %s*"%(shortName))
	os.system("mv output.csv %s.csv"%(shortName))


#def converter(filename):
#	evan = []
#	with open (filename, 'r') as fi:
#		for line in fi.readlines():
#			line = line.strip('\n').split(' ')
#			evan.append(line)
#	print(evan[0])	
	#with open('output.csv', 'w') as csv_file:
	#	writer = csv.writer(csv_file, delimiter = ',')
	#	for line in evan:
	#		writer.writerows(line)
#	df= pd.DataFrame(evan)
#	df.to_csv('%s'%(filename.split('/')[-1]+'.csv'), index = False, header = False, quoting=0)

if __name__ == '__main__':
	filename = sys.argv[1]
	converter(filename)
	print ('csv converted!')
	#vs.format(filename.split('/')[-1]+'.csv')

