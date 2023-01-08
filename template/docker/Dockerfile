FROM python:3.7

WORKDIR /code
COPY ./ .

# RUN pip3 config set global.index-url  http://mirrors.myhuaweicloud.com/pypi/web/simple \
#     && pip3 config set install.trusted-host mirrors.myhuaweicloud.com

# 如果pip安装出现异常，有可能是源的问题，换个源试试
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
RUN python -m pip install --upgrade pip

# python依赖可以直接在这里写，目前测试不带版本号没问题，或者带版本号写入requirements.txt中
RUN pip install yacs
RUN pip install pandas
RUN pip install -r requirements.txt

CMD [ "bash" ]
