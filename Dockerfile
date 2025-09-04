FROM ubuntu:24.04

RUN apt-get update -y \
      && apt-get install -y \
      python3-pip \
      python3.12-venv \
      pipx

ENV PATH="/root/.local/bin:$PATH"
RUN pipx install uv
RUN cp /root/.local/bin/uv /usr/local/bin/uv

RUN adduser --disabled-password nonroot
RUN mkdir /home/app && chown -R nonroot:nonroot /home/app
RUN mkdir -p /var/log/flask-app \
      && touch /var/log/flask-app/flask-app.err.log \
      && touch /var/log/flask-app/flask-app.out.log \
      && chown -R nonroot:nonroot /var/log/flask-app
RUN mkdir -p /home/app/logs && chown -R nonroot:nonroot /home/app/logs

WORKDIR /home/app
COPY --chown=nonroot:nonroot . .

USER nonroot

ENV VIRTUAL_ENV=/home/app/venv
ENV PATH="$VIRTUAL_ENV/bin:/usr/local/bin:$PATH"

RUN python3 -m venv $VIRTUAL_ENV
RUN uv pip install --no-cache --python $VIRTUAL_ENV/bin/python -e .
EXPOSE 9999

CMD ["python3", "-m", "src.main"]
