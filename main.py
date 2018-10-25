import numpy as np
import sklearn
import pandas as pd
from sklearn import svm
from sklearn import tree
#from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier

		
def fill(a):
	b = []
	for val in a:
		if val == 'No':
			b.append(-1)
		elif val == 'Yes':
			b.append(1)
		else:
			b.append(0)
	return b	
def end():
	print 'Thank you!'
	return
df = pd.read_csv('Downloads/bikepghpublic.csv')
fd = df

# ped and byc for the columns values of InteractPedestrian and InteractBicycle
ped = df.InteractPedestrian.values
byc =df.InteractBicycle.values
#print byc
ped1 = []
byc1 = []
for v in ped:
	if v == 'No':
		ped1.append(-1)
	elif v == 'Yes':
		ped1.append(1)
	else : 
		ped1.append(0)	
for v in byc:
	if v == 'No':
		byc1.append(-1)
	elif v == 'Yes':
		byc1.append(1)
	else : 
		byc1.append(0)		
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$		
	
#ped1 is the column matrix for pedestrian	(-1, 0, 1)			
#byc1 is the column matrix for bicycle values (-1, 0, 1)
df.ix[:, 'InteractBicycle'] = byc1
df.ix[:, 'InteractPedestrian'] = ped1
#print df.InteractPedestrian
#print df.InteractBicycle
p = df.SafetyAV.values
#print p
i = 0	
for v in p:
	if v == 'No experience':
		p[i] = 0
	elif v == '2':
		p[i] = 2
	elif v == '3':
		p[i] = 3
	elif v == '4':
		p[i] = 4
	elif v == '5':
		p[i] = 5
	elif v == '1': 
		p[i] = 1
	else:
		p[i] = 0		
	i= i + 1	

i = 0	
q = df.SafetyHuman.values
for v in q:
	if v == 'No experience':
		q[i] = '0'
	elif v == '2':
		pass
	elif v == '3':
		pass
	elif v == '4':
		pass
	elif v == '5':
		pass
	elif v == '1': 
		pass
	else:
		q[i] = '0'		
	i= i + 1	


b = []
a1 = df.ix[:, 'PayingAttentionAV'].values
for val in a1:
	if val == 'Not at all':
		b.append(0)
	elif val == 'To little extent':
		b.append(1)
	elif val == 'To some extent':
		b.append(2)
	elif val == 'To a moderate extent':
		b.append(3)		
	elif val == 'To a large extent':
		b.append(4)
	else : 
		b.append(0)		
df.ix[:, (18)] = b
rt1 = df.ix[:, 'RegulationTesting']	
RT = fill(rt1)
rs1 = df.ix[:, 'RegulationSpeed']	
RS = fill(rs1)
df.ix[:, 'RegulationTesting'] = RT
df.ix[:, 'RegulationSpeed'] = RS
# p is the new modified SafetyAV
df.ix[:, 'SafetyAV'] = p
df.ix[:, 'SaetyHuman'] = q
#print df[['InteractPedestrian', 'InteractBicycle', 'SafetyHuman', 'SafetyAV']]

target = df.ix[:, 'SafetyAV'].values
data = df.ix[:, ['InteractPedestrian', 'InteractBicycle', 'PayingAttentionAV', 'RegulationTesting', 'RegulationSpeed']].values
target = target.astype('int')

#x,y = target,scale(data)
trainf, testf, trainl, testl = tts(data,target, test_size = 0.2)
#try using svm
clf = svm.SVC(kernel = 'linear')

# learning model via svm
clf.fit(trainf, trainl)
#prediction work
pred = clf.predict(testf)

#print pred
#print testl
i = 0
score = 0
# Let's calculate the accuracy
print 'Model\t Correct values\t      accuracy'
for t in pred:
	if t == testl[i]:
		score = score + 1
	i = i + 1

print 'SVM \t\t', score, '\t\t', score * 100 / len(pred),'%'

## now try using decision tree

clf = tree.DecisionTreeClassifier()
clf.fit(trainf,trainl)
pred = clf.predict(testf)	

"""print pred
print testl"""
i = 0
score = 0
# Let's calculate the accuracy
for t in pred:
	if t == testl[i]:
		score = score + 1
	i = i + 1	
print 'Tree \t\t', score, '\t\t', score * 100 / len(pred),'%'

Line = LinearRegression()
Line.fit(trainf, trainl)
pred = Line.predict(testf)
i = 0
score = 0
for t in pred:
	if t == testl[i]:
		score = score + 1
	i = i + 1	

print 'Linear\nRegression \t', score, '\t\t', score * 100 / len(pred),'%'
#print linreg.score(data, target)
clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial')
clf.fit(trainf, trainl)
pred = clf.predict(testf)
i = 0
score = 0
for t in pred:
	if t == testl[i]:
		score = score + 1
	i = i + 1	
print 'Logistic\nRegression \t', score, '\t\t', score * 100 / len(pred),'%'	

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(trainf, trainl)
pred = knn.predict(testf)

i = 0
score = 0
for t in pred:
	if t == testl[i]:
		score = score + 1
	i = i + 1	
"""print pred
print testl	"""
print 'KNN \t\t', score, '\t\t', score * 100 / len(pred),'%'
x = []
r = 0
c = raw_input('Do you want to give your own test set? y/n\n')
if c == ('y' or 'yes' or Y):
	z = raw_input('was there a pedestrian? y/n\n')
	x.append(z)
	z = raw_input('was there bicycle ? y/n\n')
	x.append(z)
	z = raw_input('was AV paying attention? (0-4)\n')
	x.append(z)
	z = raw_input('how much the testing was regulated? (-1, 0, 1)\n')
	x.append(z)
	z = raw_input('was the speed regulated? (-1, 0, 1)\n')
	x.append(z)
else:
	r = 1
	end()
s = []
if r != 1:
	print x
i = 0
for k in x:
	if k == 'y':
		x[i] = 1
	elif k == 'n':
		x[i] = 0	
	elif k == '1' :
		x[i] = 1	
	elif k == '2' :
		x[i] = 2	
	elif k == '3':
		x[i] = 3	
	elif k == '4' :
		x[i] = 4	
	elif k == '5' :
		x[i] = 5	
	elif k == '0' :
		x[i] = 0
	elif k == '-1':	
		x[i] = -1
	i =  1 + i	
	
try:
	s = clf.predict([x])
except:
	pass
sd = ''
for k in s:
	if k == 0:
		sd = 'NO'
	if k == 1:
		sd = 'Not sure'
	if k == 2:
		sd = 'maybe'
	if k == 3:
		sd = 'moderate'
	if k == 4:
		sd = 'good'
	if k == 5:
		sd = 'Excellent'

if r != 1:
	print 'Safety of self driving car: ',sd 
