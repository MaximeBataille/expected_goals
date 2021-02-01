FROM python:3.6
COPY requirements.txt /tmp/
COPY ./app /app
WORKDIR "/app"
RUN python -m pip install -U pip
RUN pip install --trusted-host pypi.python.org -r /tmp/requirements.txt
EXPOSE 8050
ENTRYPOINT [ "python3" ]
CMD [ "python" , app.py" ]