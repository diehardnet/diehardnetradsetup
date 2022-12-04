DIEHARDNET_PATH = /home/carol/diehardnetradsetup
DATA_DIR = $(DIEHARDNET_PATH)/data
# CONFIG_NAME = resnet50_imagenet1k_v2_base
CONFIG_NAME = deeplabv3_resnet50_base

CHECKPOINTS = $(DATA_DIR)/checkpoints

YAML_FILE = $(DIEHARDNET_PATH)/configurations/$(CONFIG_NAME).yaml
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
              --generate $(ADDARGS)

test:
	$(DIEHARDNET_PATH)/$(TARGET) --iterations $(ITERATIONS) \
                  --testsamples $(TEST_SAMPLES) \
              --goldpath $(GOLD_PATH) \
              --config $(YAML_FILE) \
              --checkpointdir $(CHECKPOINTS)