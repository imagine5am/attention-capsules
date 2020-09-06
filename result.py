import sys
with open(sys.argv[1]) as eval:
	eval = eval.readlines()
model=[];char=[]; seq=[]
em = "";ec ="";es ="";
def findaccplusbrackets(inpstr,srchstr):
	strt = inpstr.find(srchstr)
	for i in range(strt,len(inpstr)):
		if inpstr[i] == ']':
			return inpstr[strt:i+1]
for e in eval:
	if 'tensorflow:Restoring parameters' in e:
		em = e;
		#model.append(e.split('-')[-1][:-1])
	elif 'CharacterAccuracy[' in e:
		ec = e;
		if 'SequenceAccuracy[' in e:
			es = findaccplusbrackets(e,'SequenceAccuracy[')
			ec = findaccplusbrackets(ec,'CharacterAccuracy[')
			#print("es,ec",es,ec)
		#print("es",es)
		if(es != ""):
			seq.append(es.split('[')[-1][:-1].replace("]",""))
			model.append(em.split('-')[-1][:-1].replace("]",""))
			char.append(ec.split('[')[-1][:-1].replace("]",""))
			ec = ""; em = ""; es ="";
		#char.append(e.split('[')[-1][:-2])
	elif 'SequenceAccuracy[' in e:
		#print(e)
		es = e;
		if 'CharacterAccuracy[' in e:
			es = findaccplusbrackets(es,'SequenceAccuracy[')
			ec = findaccplusbrackets(e,'CharacterAccuracy[')
			#print("es,ec",es,ec)
		#print("ec",ec)
		if(ec != ""):
			seq.append(es.split('[')[-1][:-1].replace("]",""))
			model.append(em.split('-')[-1][:-1].replace("]",""))
			char.append(ec.split('[')[-1][:-1].replace("]",""))
			ec = ""; em = ""; es ="";
charmax = 0.0
seqmax = 0.0
iter_seqmax = -1
iter_chrmax = -1
#print(model,char,seq)
for i in range(len(model)):
	#if(float(seq[i]) > 0.4): print("best",model[i],char[i],seq[i])
	#print("best",model[i],char[i],charmax,seq[i])
	charmax = max(float(char[i]),charmax)
	seqmax = max(float(seq[i]),seqmax)
	if seqmax == float(seq[i]):
		iter_seqmax = model[i]
	if charmax == float(char[i]):
		iter_chrmax = model[i]
	print(model[i], "{:.2f}".format(float(char[i]) * 100), "{:.2f}".format(float(seq[i]) * 100))
print("max value of charAcc is ", "{:.2f}".format(float(charmax) * 100) , "at", iter_chrmax)
print("max value of seqnAcc is ", "{:.2f}".format(float(seqmax) * 100) , "at", iter_seqmax)
