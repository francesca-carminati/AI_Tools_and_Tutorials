################################################################
# Setup
################################################################

PYTHON := python3
PROJECT_NAME := Example
MODULE_NAME := example
COMPOSE_CMD := docker-compose

# Create .env file with configs
.ONESHELL:
.env:
	@cp .env.template .env
	cat <<EOF >>.env
		COMPOSE_PROJECT_NAME=$(PROJECT_NAME)_$(shell whoami)
		PROJECT_DIR=$(PWD)
		USER_ID=$(shell id -u)
		USER_NAME=$(shell whoami)
	EOF
	cat <<EOF
		${RED}Please review the '.env' file before proceeding!${RESET}
		If you do not have a GPU, then before proceeding change the following in your '.env' file:
		- set "runtime=runc"
		- and "GPU_IDS="
	EOF
	@echo "\n====================================================================================\n"

# Change to CPU runtime if not GPU
ifeq (${GPU_IDS}, none)
        export RUNTIME := runc
else
        export RUNTIME := nvidia
endif


# Show URL for where your notebook is running
# TODO: may not work on all systems. If issue, send PR
include .env
ifneq (,$(findstring localdomain, $(shell dnsdomainname)))
        NOTEBOOK_LOCATION := localhost
else
        NOTEBOOK_LOCATION := "${shell hostname}.$(shell dnsdomainname)"
endif

# Terminal colors
BLACK   := $(shell tput -Txterm setaf 0)
RED     := $(shell tput -Txterm setaf 1)
GREEN   := $(shell tput -Txterm setaf 2)
YELLOW  := $(shell tput -Txterm setaf 3)
PURPLE  := $(shell tput -Txterm setaf 4)
MAGENTA := $(shell tput -Txterm setaf 5)
BLUE    := $(shell tput -Txterm setaf 6)
WHITE   := $(shell tput -Txterm setaf 7)
RESET   := $(shell tput -Txterm sgr0)


################################################################
# Docker compose
################################################################

.PHONY: build
build: .env
	$(COMPOSE_CMD) build

.PHONY: up
up: .env
	@echo "\n====================================================================================\n"
	@echo "\n\tYour notebook is running at: "
	@echo "\t>>> ${BLUE}http://${NOTEBOOK_LOCATION}:${JUPYTER_PORT}${RESET}\n"
	@echo "\tYour project is located at: "
	@echo "\t>>> ${GREEN}${PROJECT_DIR}${RESET}"
	@echo "\n====================================================================================\n"
	# Remove --detach to see all messages from Docker
	$(COMPOSE_CMD) up --detach

.PHONY: logs
logs: .env
	${COMPOSE_CMD} logs

.PHONY: down
down: .env
	$(COMPOSE_CMD) down

.PHONY: shell
shell: .env
	$(COMPOSE_CMD) exec jupyter-server bash


################################################################
# Code quality
################################################################

.PHONY: reformat
reformat:
	isort $(MODULE_NAME) tests
	black $(MODULE_NAME) tests

.PHONY: lint
lint:
	$(PYTHON) -m flake8 $(MODULE_NAME) tests
	$(PYTHON) -m isort --check-only $(MODULE_NAME) tests
	$(PYTHON) -m black --check $(MODULE_NAME) tests
	$(PYTHON) -m mypy $(MODULE_NAME)

.PHONY: test
test:
	PYTHONPATH=. ENVIRONMENT=local-test python -m unittest discover tests

.PHONY: build-all
build-all: lint test


################################################################
# Virtual environment
################################################################

.PHONY: env-create
env-create:
	mamba env create --file environment.yml

.PHONY: env-remove
env-remove:
	mamba env remove -n $(PROJECT_NAME)

.PHONY: env-export
env-export:
	mamba env export --from-history | grep -v "^prefix: " > environment.yml
	python ./scripts/sort_environment_file.py


################################################################
# Cookiecutter
################################################################

.PHONY: template-update
template-update:
	cookiecutter_project_upgrader --context-file cookiecutter_input.json -p True -m True
