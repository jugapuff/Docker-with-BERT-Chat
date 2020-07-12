#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

cd ../
zip -r ./deploy/ceslea.zip ./pip.conf ./requirements.txt  ./src
cd ./deploy

docker stop kor_scenario_container
docker rm kor_scenario_container
docker build -t kor_scenario_image .
docker run -it --gpus all -p 40007:50051 --name kor_scenario_container kor_scenario_image

rm -rf ./ceslea.zip
