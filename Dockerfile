FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./src /app/src

COPY ./main.py /app/main.py

COPY ./artifacts /app/artifacts

COPY ./.env /app/.env

CMD ["python", "/app/main.py"]