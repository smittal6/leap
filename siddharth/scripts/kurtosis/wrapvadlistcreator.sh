#!/bin/bash

#The aim of this script is to somehow automize the shittty process of creating a list, for the kurtosis part.
#Pliz don't kill me

inputfile=$1
nameoutputfile=$2
replacekey1="" #to be replaced in the column1
replacekey2="" #to be replaced in the colum2
replacekey3="" #isn't it obvious?

#Assume the file has three columns, essentially obtaining the raw data
cut -f1 $1 > temp1.list
cut -f2 $1 > temp2.list
cut -f3 $1 > temp3.list

#Section where actual stuff happens. Hail Sed.

#Assuming done with preparation of columns, merge them like a nigga
paste temp1.list temp2.list temp3.list > "$2"
