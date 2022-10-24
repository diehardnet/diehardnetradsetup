DIEHARDNET_PATH = /home/carol/diehardnetradsetup
DATA_DIR = $(DIEHARDNET_PATH)/data
CONFIG_NAME = c100_res44_test_02_gelu6_nans.yaml
CHECKPOINTS = $(DATA_DIR)/checkpoints

YAML_FILE = $(DIEHARDNET_PATH)/configurations/$(CONFIG_NAME)
TARGET = main.py
GOLD_PATH = $(DATA_DIR)/$(CONFIG_NAME).pt

GET_DATASET=0

ifeq ($(GET_DATASET), 1)
ADDARGS= --downloaddataset
endif

all: test generate

TEST_SAMPLES=128
ITERATIONS=10

generate:
	$(DIEHARDNET_PATH)/$(TARGET) --iterations $(ITERATIONS) \
                  --testsamples $(TEST_SAMPLES) \
              --goldpath $(GOLD_PATH) \
              --config $(YAML_FILE) \
              --checkpointdir $(CHECKPOINTS) \
              --datadir $(DATA_DIR) \
              --generate $(ADDARGS)

test:
	$(DIEHARDNET_PATH)/$(TARGET) --iterations $(ITERATIONS) \
                  --testsamples $(TEST_SAMPLES) \
              --goldpath $(GOLD_PATH) \
              --config $(YAML_FILE) \
              --checkpointdir $(CHECKPOINTS) \
              --datadir $(DATA_DIR)