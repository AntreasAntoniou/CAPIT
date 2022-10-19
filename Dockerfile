FROM ghcr.io/antreasantoniou/gate:0.3.0

SHELL ["conda", "run", "-n", "gate", "/bin/bash", "-c"]

RUN cd

ADD . /CAPMultiModal

WORKDIR /CAPMultiModal

RUN bash install_dependencies.sh

ADD entrypoint.sh /entrypoint.sh

ADD . /CAPMultiModal

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]

# The Dockerfile is pretty straightforward. It starts with the base image,
# which is the mambaforge image. Then, it installs fish and creates a conda environment.
# It clones the TALI-collector repository and installs the dependencies.
# Finally, it sets the working directory to the TALI-collector repository and sets the
# entrypoint to the entrypoint.sh script.