# image de base
FROM debian:latest

ENV VIRTUAL_ENV=/opt/venv

# on copie l'app
COPY . /app

#on va dans le dossier de l app
WORKDIR /app

# on installe les dependances avec pip
RUN apt-get update && apt-get install -y --no-install-recommends \
python3-setuptools \
python3-pip \
python3-dev \
python3-venv 

# on active venv car debian utlise pas par defaut pip
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip && pip install -r requirements.txt

# on lance le serveur flask (a pas utiliser en prod)
EXPOSE 5000
CMD gunicorn -b 0.0.0.0:5000  "run:create_app()"
