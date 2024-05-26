FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY ./src /app/src

COPY ./main.py /app/main.py

COPY ./artifacts /app/artifacts

CMD ["python", "/app/main.py"]