FROM python:3.7

RUN apk update && apk upgrade && \
    apk add --no-cache git

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/ilyakats/CUNY-DATA622-APP /usr/src/app/hw3-app

CMD [ "python", "/usr/src/app/hw3-app/main.py" ]
