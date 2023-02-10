FROM mambaorg/micromamba:1.3.0
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
RUN micromamba install -y -n base -f environment.yaml && \
  micromamba clean --all --yes

ENTRYPOINT ["fint", "$@"]
