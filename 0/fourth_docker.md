### Dockerizing 

Ein mit mlflow gespeichertes model haben wir eben mittels 'models serve' als REST service zur verfügung gestellt.

Genauso leicht ist es ein solches model in einem Docker Image zu verpacken:

	mlflow models build-docker -m ./... -n 'my_docker_model'

Einen container starten wir mit

	docker run -p 5002:8080 'my_docker_model'

und mit einem analogen curl Command ansprechen.
