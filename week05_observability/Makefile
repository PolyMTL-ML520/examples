.PHONY: up down logs ps prometheus-mode clean

up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f

ps:
	docker compose ps

prometheus-mode:
	docker compose -f docker-compose.yaml -f docker-compose.prometheus.yaml up --build

clean:
	docker compose down -v --rmi local
