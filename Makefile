.PHONY: build test notebook prepare train tune evaluate visualize clean

IMAGE = icare-cebra
RUN = docker run --rm -v "$(PWD)":/app $(IMAGE)

build:
	docker build -t $(IMAGE) .

test: build
	$(RUN) python test.py

notebook: build
	docker run --rm -p 8888:8888 -v "$(PWD)":/app $(IMAGE) \
		jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

prepare: build
	$(RUN) python scripts/prepare_data.py --nan-strategy mean

train: build
	$(RUN) python scripts/train.py --label cpc_bin --output models/cebra_cpc_bin.pt

tune: build
	$(RUN) python scripts/tune.py --label cpc_bin --output tuning/results.json

evaluate: build
	$(RUN) python scripts/evaluate.py --model models/cebra_cpc_bin.pt --label cpc_bin

visualize: build
	$(RUN) python scripts/visualize.py --model models/cebra_cpc_bin.pt --label cpc_bin

clean:
	rm -rf data/*.npz models/ tuning/ evaluation/ visualizations/
	docker rmi $(IMAGE) 2>/dev/null || true
