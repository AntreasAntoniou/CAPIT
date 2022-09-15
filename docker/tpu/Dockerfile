FROM gcr.io/deeplearning-platform-release/pytorch-xla
RUN rm -rf /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-pypy3-Linux-x86_64.sh
RUN chmod +x Mambaforge-pypy3-Linux-x86_64.sh
RUN ./Mambaforge-pypy3-Linux-x86_64.sh -bu -p /opt/conda
RUN rm Mambaforge-pypy3-Linux-x86_64.sh
RUN /opt/conda/bin/conda init bash
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh"]
CMD ["/bin/bash", "-c", "mamba activate base"]
CMD ["/bin/bash", "-c", "mamba create -n capit python=3.8"]
CMD ["/bin/bash", "-c", "mamba activate capit"]
COPY . /playground
WORKDIR /playground
RUN bash install_dependencies.sh
CMD ["/bin/bash"]