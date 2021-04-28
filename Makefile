install:
	pip install --upgrade pip &&\
	pip --no-cache-dir install -r requirements.txt
    
format:
	black *.py

test:
	python -m pytest -vv test.py

lint:
	pylint --disable=R,C hello.py

all: install format test