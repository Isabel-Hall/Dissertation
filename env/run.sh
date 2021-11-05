#!/bin/bash
docker run \
	-it \
	--rm \
	--mount type=bind,source="/home/issie/data",target=/data \
	--mount type=bind,source="/home/issie/code/py/MRI/src",target=/app \
	--name container \
	issie_pytorch:latest \
	bash
	
