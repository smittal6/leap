import re
import scipy.io.wavfile as wav
import scipy.io as sio
import numpy as np
import scipy.stats as stat

f=open('/home/neerajs/work/blurp_universe/SBC/Left.list')
f=f.read()
f=f.strip()
f=re.split('\n',f)
for i in range(len(f)):
	print(i)
	kurt_vals=[]
	[a,b]=wav.read('/home/neerajs/work/SBC/WAV_16/WAV/'+f[i]+'.wav')
	print(b.shape)
	b_part=b[0:400]
	kurt=stat.kurtosis(b_part)
	kurt_vals.append(kurt)
	overlap_behind=b[400-120:400]
	overlap_behind=np.reshape(overlap_behind,(1,120))
	start=160
	while(start+400<len(b)):
		b_part=b[start:start+160]
		b_part=np.reshape(b_part,(1,160))
		overlap_front=b[start+160:(start+160+120)]
		overlap_front=np.reshape(overlap_front,(1,120))
		b_kurt=np.hstack((overlap_behind,b_part,overlap_front))
		b_kurt=np.reshape(b_kurt,(b_kurt.shape[1],))
		kurt=stat.kurtosis(b_kurt)
		kurt_vals.append(kurt)
                overlap_behind=b[(start+160-120):start+160]
                overlap_behind=np.reshape(overlap_behind,(1,120))
		start=start+160

	kurt_vals=np.asarray(kurt_vals)
	sio.savemat('/home/neerajs/work/SBC/WAV_16/KURTOSIS/'+f[i]+'.mat',{'kurt_vals':kurt_vals})
