# Multi-tier Gram / Growth / MIC pipeline: automation commands
# Usage: `make help`

PYTHON ?= python
export MPLBACKEND = Agg
IMAGE  ?= gram-id-ast-ml:latest

.DEFAULT_GOAL := help
.PHONY: help install synthetic data analysis features train evaluate stress predict \
        all fast notebooks test lint clean clean-all docker-build docker-run docker-test

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

synthetic:  ## Regenerate the synthetic raw dataset (defaults: seed 42, 10% Gram / 3% growth artefacts)
	$(PYTHON) src/data/generate_synthetic_data.py 3.0 1.2 42 data/raw/GramID_AST_digital_microscopy.csv 0.10 0.03

data:  ## Validate the raw CSV and split train / test by isolate
	$(PYTHON) -m src.data.make_dataset

analysis: data  ## Gram reaction / ID and AST data analysis (figures 01-10)
	$(PYTHON) -m src.analysis.gram_id_analysis
	$(PYTHON) -m src.analysis.ast_analysis

features: data  ## Preprocessing + per-tier feature selection (figures 11-16)
	$(PYTHON) -m src.features.build_features

train: features  ## Train Tier 1/2 models (XGBoost, RBF-SVM, FT-Transformer) + cascades -> models/*.joblib (~6 min)
	$(PYTHON) -m src.models.train_model

evaluate: data  ## Train/Test ML + AST metrics, acceptance check, figures 19-26
	$(PYTHON) -m src.models.evaluate_model

stress: data  ## Run the 12 stress tests and the scorecard (figures 27-31, ~8 min)
	$(PYTHON) -m src.models.stress_test

predict: data  ## Run the cascades on the test isolates -> reports/predictions/
	$(PYTHON) -m src.models.predict_model

all:  ## Run the full pipeline end to end
	$(PYTHON) -m src.pipeline

fast:  ## Full pipeline without the stress tests
	$(PYTHON) -m src.pipeline --skip-stress

notebooks:  ## Execute every notebook in place (outputs saved)
	cd notebooks && for nb in 0*.ipynb; do \
	  $(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace \
	    --ExecutePreprocessor.timeout=1800 $$nb || exit 1; done

test:  ## Run the unit and integration tests
	$(PYTHON) -m pytest -q

lint:  ## Syntax-check every module (no extra linter dependency)
	$(PYTHON) -m compileall -q src tests

clean:  ## Remove generated interim/processed data and caches
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
	find data/interim data/processed -type f ! -name ".gitkeep" -delete

clean-all: clean  ## Also remove models, figures and tables
	rm -f models/*.joblib models/model_metadata.json
	rm -f reports/figures/*.png reports/tables/* reports/predictions/*.csv

docker-build:  ## Build the Docker image
	docker build -t $(IMAGE) .

docker-run:  ## Run the full pipeline inside Docker (outputs written to the mounted repo)
	docker run --rm -v "$(CURDIR)":/app $(IMAGE) make all

docker-test:  ## Run the test suite inside Docker
	docker run --rm $(IMAGE) make test
