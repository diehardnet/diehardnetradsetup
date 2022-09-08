
all: test generate

generate:
	./main.py --dnnname DiehardNetOrderICifar10 \
	          --iterations 10 \
	          --testsamples 100 \
              --batchsize 50 \
              --goldpath ./gold.pt \
              --config configurations/c100_res44_test_02_bn-relu6_base.yaml \
              --checkpointdir /home/carol/git_research/diehardnet_old/checkpoints \
              --datadir /home/carol/git_research/diehardnetradsetup/data \
              --generate

test:
	./main.py --dnnname DiehardNetOrderICifar10 \
	          --iterations 10 \
	          --testsamples 100 \
              --batchsize 50 \
              --goldpath ./gold.pt \
              --config configurations/c100_res44_test_02_bn-relu6_base.yaml \
              --checkpointdir /home/carol/git_research/diehardnet_old/checkpoints \
              --datadir /home/carol/git_research/diehardnetradsetup/data \
