FROM ghcr.io/antreasantoniou/tali-collector-dev:0.2.0

SHELL ["conda", "run", "-n", "tali-collector", "/bin/bash", "-c"]

RUN apt update
RUN apt install aptitude tree -y
RUN echo yes | pip install itables tabulate nvitop
RUN mamba update -c conda-forge ffmpeg starship -y

ENTRYPOINT ["/bin/bash"]