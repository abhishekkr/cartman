#!/usr/bin/env bash

pip install -r requirement.txt

python -m rasa_nlu.train -c data/nlu-config.yml --data data/nlu-messages.md -o models --fixed_model_name nlu --project current --verbose

python -m rasa_core.train -d data/eg-domain.yml -s data/eg-stories.md -o models/dialogue
