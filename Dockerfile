FROM pytorch/pytorch:latest
RUN pip install --default-timeout=1000 --upgrade pip

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y
RUN pip install --default-timeout=1000 --no-cache-dir  -r requirements.txt

RUN apt-get install wget
RUN wget https://drive.google.com/uc?id=1ytukthtzIYZt8IYJM9Z5d3atE3NlUh7L -O base_feature-0.0.1-py3-none-any.whl
RUN pip install base_feature-0.0.1-py3-none-any.whl

COPY . ./

EXPOSE 8005

CMD [ "python", "./app.py" ]
