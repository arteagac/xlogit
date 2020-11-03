FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install -y unzip
RUN apt-get install -y python3 python3-pip python3-venv 
RUN apt-get -y install git


# Setup python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install wheel

# Clone and move to benchmark directory
RUN git clone https://github.com/arteagac/xlogit.git
WORKDIR xlogit/examples/benchmark


# Install python requirements
RUN pip install -r requirements_bench.txt


# Install cupy
RUN python -m pip install -U setuptools pip
RUN pip install cupy-cuda110


# Install R and mlogit
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get -y install r-base
RUN apt-get install -y libxml2-dev
RUN Rscript -e "install.packages('mlogit', repos='http://cran.us.r-project.org')"
CMD ["python", "benchmark.py"]

