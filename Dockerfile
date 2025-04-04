FROM sagemath/sagemath:10.4

ARG NB_UID=1000
ARG NB_USER=sage

USER root
RUN apt update && apt install -y python3 python3-pip

USER ${NB_UID}
ENV PATH="${PATH}:${HOME}/.local/bin"
RUN pip3 install notebook
RUN ln -s $(sage -sh -c 'ls -d $SAGE_VENV/share/jupyter/kernels/sagemath') $HOME/.local/share/jupyter/kernels/sagemath-dev

# install custom package
RUN sage -pip install sage_acsv

# partially superfluous -- create separate directory to hold notebooks
WORKDIR ${HOME}/notebooks
COPY --chown=${NB_UID}:${NB_UID} . .
USER root
RUN chown -R ${NB_UID}:${NB_UID} .
USER ${NB_UID}

ENTRYPOINT []
