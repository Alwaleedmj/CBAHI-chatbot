FROM python:3.11.7

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install --upgrade pip

COPY . .

EXPOSE 8501

CMD [ "streamlit", "run", "main.py" ]