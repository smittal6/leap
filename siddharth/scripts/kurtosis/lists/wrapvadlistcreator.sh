#!/bin/bash

#The aim of this script is to somehow automize the shittty process of creating a list, for the kurtosis part.
#Pliz don't kill me

#inputfile="../lists/wrapvadsingletrain.list"
nameoutputfile="../lists/fwrapvadoverlaptrain.list"
repwhat1="/home/neerajs/siddharth/feats/train/overlap/mfcc/"
repwhat2="/home/neerajs/siddharth/vad_overlap/train/"
repwhat3="/home/neerajs/siddharth/feats/train/aftervado/"
replacekey1="/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/overlap/" #to be replaced in the column1
replacekey2="/home/smittal/Desktop/coding/leap/siddharth/vad_overlap/train/vad_overlap22/" #to be replaced in the column2
replacekey3="/home/smittal/Desktop/coding/leap/siddharth/kurtosis/train/aftervado/" #isn't it obvious?

#Assume the file has three columns, essentially obtaining the raw data
cut -f1 $1 > temp1.list
cut -f2 $1 > temp2.list
cut -f3 $1 > temp3.list

#Section where actual stuff happens. Hail Sed.
sed -i "s|$repwhat1|$replacekey1|g" temp1.list
sed -i "s|$repwhat2|$replacekey2|g" temp2.list
sed -i "s|$repwhat3|$replacekey3|g" temp3.list
#Assuming done with preparation of columns, merge them like a nigga
paste temp1.list temp2.list temp3.list > temp7.list
cat temp7.list | sed 's/htk/mat/g' > $nameoutputfile
rm temp*
