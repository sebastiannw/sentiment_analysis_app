setup:
	python3 -m venv ~/.sentiment_analysis_app

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt
    
format:
	black *.py

test:
	python -m pytest -vv test.py

lint:
	pylint --disable=R,C hello.py

all: install format test