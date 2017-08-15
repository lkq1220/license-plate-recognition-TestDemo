#!/bin/bash

#批处理img文件夹下的图片
for file in ./img/*.jpg
do
	echo "Processing $file file..."
    ./CharsSegment $file
done 