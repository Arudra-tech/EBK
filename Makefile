# EBK dev commands. `make dev` runs everything (requires MongoDB running).

.PHONY: dev sim server dash mongo mongo-docker stop \
        harness harness-preflight engines accuracy subset clocks

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
	-pkill -f "uvicorn harness.main:app"

# --- hardware harness (runs ON the GB10 / Jetson, in its own venv; see harness/README.md) ---
PY ?= python3
# ultralytics otherwise runs `pip install ...` on its own the first time it exports an
# engine — on a slow/captive network that "hangs" for many minutes. Fail fast instead.
export YOLO_AUTOINSTALL = false

harness:          ## real workload service on :8100 (one worker — it owns the GPU)
	$(PY) -m uvicorn harness.main:app --host 0.0.0.0 --port $${HARNESS_PORT:-8100} --workers 1

harness-preflight: ## the 10:05 checks; exit code = number of failures
	$(PY) tools/preflight.py

engines:          ## build TensorRT engines for this device (skips existing)
	$(PY) tools/build_engines.py

accuracy:         ## precompute the accuracy table for this device
	$(PY) tools/precompute_accuracy.py

subset:           ## build the COCO eval subset (laptop, night before)
	$(PY) harness/data/make_subset.py

clocks:           ## lock clocks (needs sudo)
	sudo bash tools/lock_clocks.sh
