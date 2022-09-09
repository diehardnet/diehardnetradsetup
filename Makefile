
all: test generate

TEST_SAMPLES=1024
ITERATIONS=10

generate:
	./main.py --iterations $(ITERATIONS) \
	          --testsamples $(TEST_SAMPLES) \
              --goldpath ./gold.pt \
              --config configurations/c100_res44_test_02_bn-relu6_base.yaml \
              --checkpointdir /home/carol/git_research/diehardnet_old/checkpoints \
              --datadir /home/carol/git_research/diehardnetradsetup/data \
              --generate

test:
	./main.py --iterations $(ITERATIONS) \
	          --testsamples $(TEST_SAMPLES) \
              --goldpath ./gold.pt \
              --config configurations/c100_res44_test_02_bn-relu6_base.yaml \
              --checkpointdir /home/carol/git_research/diehardnet_old/checkpoints \
              --datadir /home/carol/git_research/diehardnetradsetup/data