# Use the miniconda3 as base image
FROM continuumio/miniconda3:23.5.2-0

RUN apt-get update
RUN apt-get install -y build-essential
COPY ./environment.yml /tmp/environment.yml
# Update conda
RUN conda env create -f  /tmp/environment.yml
RUN pip cache purge

COPY . /data/workspace/pytorch/examples/fast_neural_style

# Set the working directory to /app
WORKDIR /data/workspace/pytorch/examples/fast_neural_style

# Run any command you want when the container launches
CMD [ "/bin/bash", "-c", "source activate fast_ns && python neural_style/neural_style_server.py" ]

#docker build -t animeganv2:v1.0 .

