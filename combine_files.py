#import glob
import sys
import re
import os
#non_spam_file_name = glob.glob("/Users/franciscosantos/Documents/SPAM_Project/Data/enron1_train/train/ham/*.txt")
#spam_file_name = glob.glob("/Users/franciscosantos/Documents/SPAM_Project/Data/enron1_train/train/spam/*.txt")
table = []
path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/enron1_train/train/ham/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "ham,"+clean
    table.append(clean)
path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/enron4_train/train/ham/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "ham," + clean
    table.append(clean)

path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/train/ham/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "ham," + clean
    table.append(clean)

path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/enron1_train/train/spam/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "spam," + clean
    table.append(clean)


path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/enron4_train/train/spam/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "spam," + clean
    table.append(clean)

path = sys.argv[0]
path =path.replace("combine_files.py",r"Data/train/spam/")
#path+= "\\"+r"enron1_train/train/ham/"
filesList = os.listdir(path)
for i in filesList:
    file_name = path+i
    file= open(file_name,'r', encoding="utf8", errors = 'replace')
    clean =""
    for line in file:
        line = line.strip().replace("\n"," ")
        clean+=line
    clean = "spam," + clean
    table.append(clean)

write_file = open("train.txt", "w")
for i in table:
    ln = i+"\n"
    write_file.write(ln)
write_file.close()