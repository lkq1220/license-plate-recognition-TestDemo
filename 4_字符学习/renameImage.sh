#!/bin/bash

#重命名img文件夹下的图片
for directory in ./img/*
do
	if [ -d $directory ]
	then
		i=0
		for file in $directory/*.jpg
		do
			echo "Processing $file file..."
			mv $file $directory/$i.jpg
			i=`expr $i + 1`
		done
	fi
done 
