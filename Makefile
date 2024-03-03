#############################
## Front Matter            ##
#############################

.PHONY: help
.DEFAULT_GOAL := help

#############################
## Targets                 ##
#############################

## build the docker image
build-docker:
	docker build -t akws/lms4text2sparql .

## run the copyu dataset
run-coypu:
	docker run --rm -it --init --ipc=host --user="$(id -u):$(id -g)" --volume="$(shell pwd):/app" akws/lms4text2sparql python3 train.py --dataset coypu

## run the orga dataset
run-orga:
	docker run --rm -it --init --ipc=host --user="$(id -u):$(id -g)" --volume="$(shell pwd):/app" akws/lms4text2sparql python3 train.py --dataset orga

#############################
## Help Target             ##
#############################

## Show this help
help:
	@printf "Available targets:\n\n"
	@awk '/^[a-zA-Z\-\_0-9]+:/ { \
		helpMessage = match(lastLine, /^## (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 3, RLENGTH); \
			printf "  %-40s%s\n", helpCommand, helpMessage; \
		} \
	} \
	{ lastLine = $$0 }' $(MAKEFILE_LIST) | sort
