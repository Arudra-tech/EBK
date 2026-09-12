# EBK dev commands. `make dev` runs everything (requires MongoDB running).

.PHONY: dev sim server dash mongo mongo-docker stop

mongo:            ## start MongoDB (macOS / Homebrew)
	brew services start mongodb-community

mongo-docker:     ## start MongoDB (Docker — teammates / GB10)
	docker compose up -d mongo

sim:              ## GB10 workload simulator on :8100
	uv run uvicorn simulator.main:app --port 8100

server:           ## EBK server (API + WS + watcher) on :8000
	uv run uvicorn server.app:app --port 8000

dash:             ## dashboard on :5173
	cd dashboard && npm run dev

dev:              ## run simulator + server + dashboard together
	$(MAKE) -j3 sim server dash

stop:
	-pkill -f "uvicorn simulator.main:app"
	-pkill -f "uvicorn server.app:app"
