FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.doubanio.com/simple/

COPY . .
#RUN mv bin/phantomjs /usr/local/bin/

#CMD ["python", "-m", "bb8"]