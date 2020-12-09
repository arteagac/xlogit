FROM nvidia/cuda:11.0-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y curl unzip ca-certificates software-properties-common wget
RUN apt-get install -y libssl-dev libcurl4-openssl-dev

RUN wget https://raw.githubusercontent.com/arteagac/xlogit/master/examples/benchmark/requirements_python.txt
RUN wget https://raw.githubusercontent.com/arteagac/xlogit/master/examples/benchmark/requirements_r.R

# Setup R environment and install mlogit and apollo
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get update
RUN apt-get -y install r-base
RUN apt-get install -y libxml2-dev
RUN Rscript requirements_r.R


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    sh ~/miniconda.sh -b -p ~/conda && rm ~/miniconda.sh && ~/conda/bin/conda install -y python=3.7.9 && \
    ln -s ~/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". ~/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc 

RUN ln -s ~/conda/bin/conda /usr/bin/conda
RUN conda init bash && . ~/.bashrc \
    && pip install -U setuptools pip && pip install numpy \
    && pip install -r requirements_python.txt # && conda install cudatoolkit==11.0.221

COPY . xlogit
WORKDIR xlogit/examples/benchmark

RUN printf '#!/bin/bash --login\nset -e\nconda activate base\n$@' >> entry.sh && chmod 777 entry.sh

ENTRYPOINT ["./entry.sh"]
CMD ["python", "benchmark.py", "mini"]

