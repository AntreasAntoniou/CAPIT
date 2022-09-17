FROM gcr.io/deeplearning-platform-release/pytorch-xla
FROM condaforge/mambaforge:latest as conda

SHELL ["/bin/bash", "-c"]

RUN conda init bash
RUN conda create -n capit python=3.8 -y
RUN echo "conda activate capit" >> ~/.bashrc

RUN git clone https://github.com/AntreasAntoniou/CAPMultiModal.git


SHELL ["conda", "run", "-n", "capit", "/bin/bash", "-c"]

RUN cd CAPMultiModal  \
    && bash install_dependencies.sh

CMD ["bin/bash"]