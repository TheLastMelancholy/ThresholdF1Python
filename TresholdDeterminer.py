import numpy as np

thresholdIOU = 0.5

#FUNCTION PARSING DATASET BY SINGLE LINES

class match():
	def __init__(self, name, confidence, isPositive):
		self.name=name
		self.confidence=float(confidence)
		self.isPositive=bool(isPositive)
	def __repr__(self):
		return "{0:10} is {1} wih confidence {2}".format(self.name, self.isPositive, self.confidence)
		
class allMatches():
	def __init__(self):
		self.matches = {}
	def add(self, m):
		if m.name in self.matches:
			self.matches[m.name].append(m)
		else:
			self.matches[m.name]=[]
			self.matches[m.name].append(m)

	def __getitem__(self, key):
		return list(self.matches.values())[key]

	def __len__(self):
		return len(list(self.matches.values()))

	def getKeyName(self, key):
		return list(self.matches.keys())[key]

	def F1(self, key, threshold):
		#recall     = true positives/(true positives + false negatives)
		#precission = true positives/(true positives + false positives)
		#f1 = 2*(precission*recall)/(precission+recall)
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

#FUNCTIONS PARSING DATA OF EACH ITERATION
def IoU(sq1, sq2): 
	IL = int(max(int(sq1[0]), int(sq2[0])))
	IT = int(max(int(sq1[1]), int(sq2[1])))
	IR = int(min(int(sq1[2]), int(sq2[2])))
	IB = int(min(int(sq1[3]), int(sq2[3])))

	IS = max(0, IR-IL)*max(0, IB-IT)

	sq1S = (int(sq1[2])-int(sq1[0]))*(int(sq1[3])-int(sq1[1]))
	sq2S = (int(sq2[2])-int(sq2[0]))*(int(sq2[3])-int(sq2[1]))

	IOU = float(IS)/float(sq1S+sq2S-IS)
	#print(IOU)
	return IOU

def checkIoU(sq1, sq2):
	#print(sq1)
	#print(sq2)
	if(IoU(sq1, sq2) >=thresholdIOU):
		return True
	return False



def findUniqueElements(record):
	unique = {}
	for r in record:
		if not r[0] in unique:
			if not (r[0] =='\n'):
				unique[r[0]]=r[1:]
	return unique

def processRecord(record):
	u = findUniqueElements(record[0])
	u1 = findUniqueElements(record[1])

	matches = []

	for key in u:
		if key in u1:
			el1=u[key]
			el2=u1[key]
			if(checkIoU(el1, el2)):
				m = match(key, el2[4], True)
			else:
				m = match(key, el2[4], False)
			matches.append(m)
		else:
			m = match(key, 0.0, True) # not shure
			matches.append(m)
	for key in u1:
		if not key in u:
			m = match(key, u1[key][4], False)
			matches.append(m)
	return matches



def floatRange(b, e, s):
	i = b
	while(i<=e):
		yield i
		i+=s

def iterateMaxF1(key, m):
	maxF1 = 0.0
	maxTr = 0.0
	for i in floatRange(0.01, 0.99, 0.01):
		try:
			f = m.F1(key, i)
		except(ZeroDivisionError):
			continue
		if(f >= maxF1):
			maxF1, maxTr = f, i
	return (maxF1, maxTr)


m = allMatches()
gen = parseDataset("detection_val_log.txt")
#for d in range(20000):
for obj in gen:
	rec = processRecord(obj)
	for r in rec:
		#print(r)
		m.add(r)

for i in range(len(m)):
	res = iterateMaxF1(i, m)
	print("{0} optimal threshold is {1:2f} with F1 of {2}".format(m.getKeyName(i), res[1], res[0]))

