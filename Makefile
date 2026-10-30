.PHONY: install run docker-build docker-run deploy

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -r requirements.txt

run:
	python wsgi.py

docker-build:
	docker build -t tesis-forecast:latest .

docker-run:
	docker run -p 8080:8080 tesis-forecast:latest

# Requiere tener gcloud configurado y autenticado
deploy:
	gcloud run deploy tesis-forecast \
		--source . \
		--project tesis-usdmxn-2026 \
		--region us-central1 \
		--allow-unauthenticated \
		--memory 2Gi \
		--cpu 1 \
		--timeout 300 \
		--min-instances 0 \
		--max-instances 3 \
		--session-affinity \
		--update-env-vars GIT_COMMIT=$$(git rev-parse --short HEAD),BUILD_TIME=$$(date -u +%Y-%m-%dT%H:%MZ)
