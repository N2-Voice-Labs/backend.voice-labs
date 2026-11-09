# Local Docker stack helpers. Config comes from ./.env (cp .env.example .env).
COMPOSE ?= docker compose

.PHONY: up down logs ps migrate reset test public public-down
.DEFAULT_GOAL := ps

up: ## build images and start the whole stack in the background (local-livekit profile by default)
	@p="$$COMPOSE_PROFILES"; \
	[ -n "$$p" ] || p=$$(grep -E '^COMPOSE_PROFILES=' .env 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'"'"' '); \
	COMPOSE_PROFILES="$${p:-local-livekit}" $(COMPOSE) up -d --build

down: ## stop and remove containers (keeps the database volume)
	$(COMPOSE) --profile '*' down

# Public demo: Cloudflare Tunnel + LiveKit Cloud (config from .env, see .env.public.example).
# Values are read from .env without printing them.
public: ## start the stack with the Cloudflare tunnel (needs CLOUDFLARE_TUNNEL_TOKEN, wss:// LIVEKIT_URL)
	@set -a; [ -f .env ] && . ./.env; set +a; \
	if [ -z "$$CLOUDFLARE_TUNNEL_TOKEN" ]; then \
	  echo "make public: CLOUDFLARE_TUNNEL_TOKEN is not set. cp .env.public.example .env and fill it (Cloudflare Zero Trust > Networks > Tunnels)."; exit 1; fi; \
	case "$$LIVEKIT_URL$$LIVEKIT_PUBLIC_URL" in *FILL-ME*) echo "make public: replace the FILL-ME placeholders in .env with your LiveKit Cloud URL."; exit 1;; esac; \
	case "$$LIVEKIT_URL" in wss://*) ;; \
	  *) echo "make public: LIVEKIT_URL must start with wss:// (your LiveKit Cloud project URL); got '$$LIVEKIT_URL'."; exit 1;; esac; \
	case "$$LIVEKIT_PUBLIC_URL" in wss://*) ;; \
	  *) echo "make public: LIVEKIT_PUBLIC_URL must start with wss:// (same LiveKit Cloud URL); got '$$LIVEKIT_PUBLIC_URL'."; exit 1;; esac; \
	COMPOSE_PROFILES=tunnel $(COMPOSE) up -d --build

public-down: ## stop the public demo (tunnel included); keeps the database volume
	$(COMPOSE) --profile '*' down

logs: ## follow logs (make logs S=platform for one service)
	$(COMPOSE) logs -f --tail=100 $(S)

ps: ## show service status
	$(COMPOSE) ps -a

migrate: ## re-run Alembic migrations against the running database
	$(COMPOSE) run --rm migrate

reset: ## DESTROY containers and the database volume (asks first)
	@printf 'This deletes the Postgres volume and all local data. Continue? [y/N] ' && read ans && [ "$$ans" = y ] || [ "$$ans" = Y ] || { echo aborted; exit 1; }
	$(COMPOSE) --profile '*' down -v

test: ## run the three pytest suites on the host via uv
	uv run --directory platform --locked --extra test pytest -q
	uv run --directory runtime --locked --extra voice --extra test pytest -q
	uv run --directory evals --locked --extra test pytest -q
