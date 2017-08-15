#!/bin/bash

#重命名img文件夹下的图片
i=0
for file in ./img/*.jpg
do
	echo "Processing $file file..."
    mv $file ./img/$i.jpg
    i=`expr $i + 1`
done
