import numpy as np

thresholdIOU = 0.5


class match():
	'''
	Class storing infromation about recognised (or not recognised) object
	It stores recognition object name
			  					 confidence level of prediction
			  					 and boolean value indicating whether prediction match real object or not
	'''
	def __init__(self, name, confidence, isPositive):
		self.name=name
		self.confidence=float(confidence)
		self.isPositive=bool(isPositive)
	def __repr__(self):
		return "{0:10} is {1} wih confidence {2}".format(self.name, self.isPositive, self.confidence)
		

class dSetStatistics():
	'''
	Class stores data about all objects in dataset and about all predictions of this objects 
	
	'''
	def __init__(self):
		self.matches = {}
	def add(self, m):
		'''
		Append recognition record to existing object-class or create new 
		'''
		if not m.name in self.matches:
			self.matches[m.name]=[]
		self.matches[m.name].append(m)


	def __getitem__(self, key):
		return list(self.matches.values())[key]

	def __len__(self):
		return len(list(self.matches.values()))

	def getKeyName(self, key):
		return list(self.matches.keys())[key]

	def F1(self, key, threshold):
		'''
		calculate F1 score for all predictions of given class with given threshold where

		recall     = true positives/(true positives + false negatives)
		precission = true positives/(true positives + false positives)
		F1         = 2*(precission*recall)/(precission+recall)
		'''
		tp = 0.0
		fp = 0.0
		fn = 0.0

		dataset = self[key]

		for el in dataset:
			if(    el.isPositive and el.confidence >  threshold): tp+=1.0
			if(    el.isPositive and el.confidence <= threshold): fp+=1.0
			if(not el.isPositive and el.confidence >  threshold): fn+=1.0

		recall     = tp/(tp+fn)
		precission = tp/(tp+fp)

		score = 2.0*(precission*recall)/(precission+recall)

		return score

#FUNCTION PARSING DATASET BY SINGLE LINES

def parseRecord(record):
	resultsData = []
	results = record.split(";")
	for r in results:
		resultsData.append(r.split(","))
	return resultsData


def parseLineOfData(line):
	splitted = line.split("--")
	testResults=[]
	for s in splitted:
		testResults.append(parseRecord(s))
	return testResults

def parseDataset(file):
	reader=open(file, "r")
	data = reader.readlines()
	for d in data:
		yield parseLineOfData(d)

#FUNCTION PARSING DATASET BY SINGLE LINES

#FUNCTIONS PROCESSING DATA OF EACH ITERATION
def IoU(sq1, sq2): 
	mx = lambda a, b: int(max(int(a), int(b)))
	mn = lambda a, b: int(min(int(a), int(b)))
	sq = lambda a   :  (int(a[2])-int(a[0]))*(int(a[3])-int(a[1]))

	ILef = mx(sq1[0], sq2[0])
	ITop = mx(sq1[1], sq2[1])
	IRig = mn(sq1[2], sq2[2])
	IBot = mn(sq1[3], sq2[3])

	IS = max(0, IRig-ILef)*max(0, IBot-ITop)

	sq1S = sq(sq1)
	sq2S = sq(sq2)

	IOU = float(IS)/float(sq1S+sq2S-IS)

	return IOU

def checkIoU(matchGuess):
	if(IoU(matchGuess[0], matchGuess[1]) >= thresholdIOU):
		return True
	return False



def packInDict(record):
	packed = {}
	for r in record:
		if not (r[0] =='\n'):
			packed[r[0]]=r[1:]
	return packed

def processLine(record):
	'''
	If object were predicted (record exists both in left and right parts of recognition report)
	Check IoU value for prediction and mark match as positive or negative
	If object exist only in left  part of report ~ mark as positively predicted with 0.0 confidence (Not shure about it)
	If object exist only in right part of report ~ mark as negatively predicted
	'''
	LData = packInDict(record[0])
	RData = packInDict(record[1])

	matches = []

	for key in LData:
		if key in RData:
			matchGuess = (LData[key], RData[key])
			if(checkIoU(matchGuess)):
				m = match(key, matchGuess[1][4], True)
			else:
				m = match(key, matchGuess[1][4], False)
		else:
			m = match(key, 0.0, True)
		matches.append(m)

	for key in RData:
		if not key in LData:
			m = match(key, RData[key][4], False)
			matches.append(m)
	return matches

#FUNCTIONS PROCESSING DATA OF EACH ITERATION

#FUNCTIONS CALCULATING OPTIMAL F1
def floatRange(b, e, s):
	i = b
	while(i<=e):
		yield i
		i+=s

def iterateMaxF1(key, m):
	maxF1 = 0.0
	maxTr = 0.0
	for tr in floatRange(0.01, 0.99, 0.01):
		try:
			F1 = m.F1(key, tr)
		except(ZeroDivisionError):
			continue
		if(F1 >= maxF1):
			maxF1, maxTr = F1, tr
	return (maxF1, maxTr)

#FUNCTIONS CALCULATING OPTIMAL F1

#PROGRAM ENTRY POINT

dbStats = dSetStatistics()
dataReader = parseDataset("detection_val_log.txt")
for parsedLine in dataReader:
	processedLine = processLine(parsedLine)
	for processedRecord in processedLine:
		dbStats.add(processedRecord)

for detectionClass in range(len(dbStats)):
	optimum = iterateMaxF1(detectionClass, dbStats)
	print("{0:10} optimal threshold is {1:3f} with F1 of {2}".\
		format(dbStats.getKeyName(detectionClass), optimum[1], optimum[0]))

