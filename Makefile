.PHONY: train

train:
	@echo "Cleaning outputs folder..."
	@rm -rf outputs
	@mkdir -p outputs
	@echo "Running training script..."
	@uv run src/scripts/train.py

