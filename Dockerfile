FROM python:3.7.3-stretch

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . app.py /code/predict.py /code/utils.py /code/neural_net.py /data/word2int /models/Sentiment_LSTM_dictionary /app/

# Install packages from requirements.txt
# hadolint ignore=DL3013
RUN pip install --upgrade pip &&\
    pip install --trusted-host pypi.python.org -r requirements.txt &&\
    python -m nltk.downloader punkt