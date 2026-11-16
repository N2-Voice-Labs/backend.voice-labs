COMPOSE ?= docker compose
PLATFORM_URL ?= http://localhost:8000
PYTHON ?= python
.DEFAULT_GOAL := up
.PHONY: up down logs seed test
up:
	$(COMPOSE) up -d --build
down:
	$(COMPOSE) --profile '*' down
logs:
	$(COMPOSE) logs -f --tail=100 $(S)
seed:
	@test -n "$(API_KEY)" || { echo 'Set API_KEY to a Platform admin key'; exit 1; }
	$(PYTHON) scripts/seed_demo_agent.py --platform-url "$(PLATFORM_URL)" --api-key "$(API_KEY)"
test:
	cd platform && $(PYTHON) -m pytest -q
	cd runtime && $(PYTHON) -m pytest -q
	cd evals && $(PYTHON) -m pytest -q
