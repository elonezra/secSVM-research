import staticAnalyzer
import sys
import os
from glob import glob
path=sys.argv[1]
m_or_b = sys.argv[2] # write in the command after the apps location 1 for malisous or 0 for the other
result=os.listdir(path)
str_path = str(sys.argv[1])+"/data_set/lable.json"
lable = ""

last=len(result)-1
for i in result:
        ind=result.index(i)
        if not(i.endswith(".apk")):
                continue
        try:
                staticAnalyzer.run(path+"/"+ i, str(sys.argv[1]) +"/",ind,last,str(sys.argv[1]))
                lable += m_or_b
                lable += ","
        except Exception as e:
                print(e)
                print("file "+str(ind)+" failed")
                continue
is_lable_exist = False
if os.path.exists(str_path):
	is_lable_exist = True
	with open(str_path, 'rb+') as f:
		f.seek(-1, 2)
		f.truncate()
	f.close()
with open(str_path,"a+") as r:
	if is_lable_exist:
		r.write(","+lable[:-1]+"]")
	else:
		r.write("[" + lable[:-1]+"]")
r.close()

