FROM mambaorg/micromamba:1.3.0
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
RUN micromamba install -y -n base -f environment.yml && \
  micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install .
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "./entrypoint.sh"]
