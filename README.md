# Video Analytics using Docker Containers
Video analytics for vehicle using tensorRT and API serving within a docker container.
This demonstrates dockerized REST APIs for CNN model using Redis. 
 

## Prerequisites
* docker-compose(2.3 version)/Nvidia-docker-compose 
* Python 3 #dependency met within docker container
* Keras (with tensorflow-gpu preffered) #dependency met within docker container
* Rest API, Redis/Flask #dependency met within docker container
* Nvidia-tensorRT #dependency met within docker container
* Opencv,Numpy #dependency met within docker container

## How to run
We will be using docker-compose-up/down to host the REST API within the docker container.

To start the API,
_sudo docker-compose up_

Use the following curl -X POST command to test the model. You will get a json response relevant to your call.
@image/location/image.jpg should be the actual image. Port can be changed in the docker file/server.py. 
"_curl -X POST -F image=@image/location/image.jpg 'http://localhost:5000/predict'_"


## Models in the Project
You can replace the models with the finetuned models on your dataset. You can swap the model with your model in _infer-iva.py_.  
 * Vehicle Make,Color and Type - Resnet model pretrained in tensorRT
 * ANPR - WPOD Net for detection of licence plate and yolo(darknet) for character recognition

## 
