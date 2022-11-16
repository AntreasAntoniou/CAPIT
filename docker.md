To build a docker container suitable for experimentation with this repository, please follow the following instructions.

1. git clone CAPMultiModal-docker-containers
2. sudo docker build -t capit-tpu docker/tpu
3. sudo docker run -it --name capit-tpu -v /mnt/:/mnt/ capit-tpu