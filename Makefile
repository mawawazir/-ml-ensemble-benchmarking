.PHONY: install benchmark test lint clean add-model format help

PYTHON := python3
PIP := pip

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install project dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

benchmark: ## Run the full benchmark across all models (Random Forest, XGBoost, SVM)
	$(PYTHON) -m src.models.benchmark

test: ## Run the test suite with pytest
	pytest tests/ -v --tb=short

lint: ## Run flake8 and check formatting with black
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/

format: ## Auto-format code with black
	black src/ tests/

clean: ## Remove build artifacts, caches, and generated reports
	rm -rf build/ dist/ *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf reports/figures/*.png
	rm -f reports/benchmark_results.json

add-model: ## Scaffold a new model (usage: make add-model NAME=my_model)
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make add-model NAME=<model_name>"; \
		exit 1; \
	fi
	@echo "Creating src/models/$(NAME).py ..."
	@cp src/models/random_forest.py src/models/$(NAME).py
	@echo "Creating configs/$(NAME).yaml ..."
	@cp configs/random_forest.yaml configs/$(NAME).yaml
	@echo "Scaffolding complete. Edit src/models/$(NAME).py and configs/$(NAME).yaml to implement your model."
