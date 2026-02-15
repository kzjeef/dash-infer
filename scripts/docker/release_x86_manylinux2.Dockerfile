FROM quay.io/pypa/manylinux2014_x86_64

RUN yum install openssl-devel curl-devel wget python3-devel -y --nogpgcheck
RUN yum install -y bzip2 epel-release

ARG PY_VER=3.8

RUN curl -LO https://github.com/Kitware/CMake/releases/download/v3.27.9/cmake-3.27.9-linux-x86_64.sh \
    && bash ./cmake-3.27.9-linux-x86_64.sh --skip-license --prefix=/usr

RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -p /miniconda -b \
    && rm -f Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

##########################################################################
# conda uses default channels; set mirrors if needed for China builds:
#   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
##########################################################################

RUN conda clean -i -y && conda config --show channels && conda create -y --name ds_py python==${PY_VER} && conda update -n base conda
SHELL ["conda", "run", "-n", "ds_py", "/bin/bash", "-c"]
RUN echo "source activate ds_py" >> /root/.bashrc && source /root/.bashrc

##########################################################################
# pip uses default PyPI; set mirrors if needed for China builds:
#   pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
##########################################################################

RUN yum install -y atlas-devel
RUN pip3 install auditwheel

##########################################################################
# github action requirements
##########################################################################
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash \
    && yum install git-lfs -y

RUN yum install -y libtool flex

RUN wget "ftp://ftp.gnu.org/gnu/automake/automake-1.15.1.tar.gz" && \
    tar -xvf automake-1.15.1.tar.gz && \
    cd automake-1.15.1 && ./configure --prefix=/usr/local/ && make -j && make install && \
    cd .. && rm -rf automake-1.15.1.tar.gz automake-1.15.1


WORKDIR /root/
