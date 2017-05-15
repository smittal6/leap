#The aim of this script is to output a matrix, which forms the complete training data
#Thus, we'll have the features, as well as the labels. 
#Single speaker speech, label is 0, Overlap speech label is 1
import htkmfc as htk
import numpy as np
import os

#Edit this
os.chdir('lpcres')

reader=htk.open('overlap.htk')
data=reader.getall()
reader=htk.open('overlaplabels.htk')
labels=reader.getall()
Data=np.hstack((data,labels))

reader=htk.open('single.htk')
data=reader.getall()
reader=htk.open('singlelabels.htk')
labels=reader.getall()
Data=np.vstack((Data,np.hstack((data,labels))))

### IT SEEMS THE FOLLOWING STUFF IS NOT REQUIRED FOR CREATING INPUT MATRIX"
"""
reader=htk.open('overlap_test.htk')
data=reader.getall()
reader=htk.open('overlap_testlabels.htk')
labels=reader.getall()
Data=np.vstack((Data,np.hstack((data,labels))))

reader=htk.open('test.htk')
data=reader.getall()
reader=htk.open('testlabels.htk')
labels=reader.getall()
Data=np.vstack((Data,np.hstack((data,labels))))
"""
#Not sure about the following parameters
writer=htk.open('lpcres.htk',mode='w',veclen=20)
writer.writeall(Data)

