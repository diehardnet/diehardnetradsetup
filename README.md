# Radiation setup for DieHardNet
Radiation Setup to evaluate DieHardNet in a realistic way

# Getting started

## Requirements

- You need to create a symlink or copy the 
[pythorch_scripts](https://github.com/diehardnet/diehardnet/tree/main/pytorch_scripts) 
folder from the original [diehardnet](https://github.com/diehardnet) main repository. 

- For the beam experiments, you will need the scripts from [radhelper](https://github.com/radhelper) repositories 
to control the boards inside the beamline
  - You must have [libLogHelper](https://github.com/radhelper/libLogHelper) 
  installed on the host that controls the GPU and a socket server set up outside the beam room. 
  You can use [radiation-setup](https://github.com/radhelper/radiation-setup) as a socket server.
- For fault simulations, you can use the official version of 
[NVIDIA Bit Fault Injector](https://github.com/NVlabs/nvbitfi) (works until Volta micro-architecture) or 
the version
  we updated for [Ampere evaluations](https://github.com/fernandoFernandeSantos/nvbitfi/tree/new_gpus_support).

## Running an experiment

Here are the main parameters necessary for reliability evaluations. 
These can be used for fault injection using NVBitFI or for radiation experiments.

```bash
usage: main.py [-h] [--config FILE] [--iterations ITERATIONS] [--testsamples TESTSAMPLES] [--generate] [--disableconsolelog] [--goldpath GOLDPATH] [--checkpointdir CHECKPOINTDIR]

optional arguments:
  -h, --help            show this help message and exit
  --config FILE         YAML config file specifying default arguments.
  --iterations ITERATIONS
                        Iterations to run forever
  --testsamples TESTSAMPLES
                        Test samples to be used in the test.
  --generate            Set this flag to generate the gold
  --disableconsolelog   Set this flag disable console logging
  --goldpath GOLDPATH   Path to the gold file
  --checkpointdir CHECKPOINTDIR Path to checkpoint dir
```

### Setting up the experiment

A complete line to execute the main.py script can look like this:

```bash
 PYTHONPATH=<path to libLogHelper>/libLogHelper/build ./main.py --iterations 10000 --testsamples 1024 --config ./configurations/c10_res44_test_01_bn-relu_base.yaml --checkpointdir ./data/checkpoints --goldpath ./data/c10_res44_test_01_bn-relu_base.pt --generate
```

 We created a configure.py script to avoid manually generating all the possible configurations for the evaluation. 
 The script automatically generates the golden outputs and JSON files with the command line to execute each config. 

To run [configure.py](https://github.com/diehardnet/diehardnetradsetup/blob/main/configure.py), 
set the desired parameters in the first lines of code as shown in the example below:

```python
ALL_DNNS = configs.DIEHARDNET_CLASSIFICATION_CONFIGS
CONFIG_FILE = "/etc/radiation-benchmarks.conf"
ITERATIONS = int(1e12)
TEST_SAMPLES = {
    **{k: 128 * 8 for k in configs.DIEHARDNET_CLASSIFICATION_CONFIGS},
}
```

Then just run the script:
```bash
./configure.py
```
You can also use specific parameters for the script, like bellow:
```bash
usage: configure.py [-h] [--testjsons TESTJSONS] [--downloadmodels] [--downloaddataset]

Configure a setup

optional arguments:
-h, --help            show this help message and exit
--testjsons TESTJSONS
                    How many seconds to test the jsons, if 0 (default) it does the configure
--downloadmodels      Download the models
--downloaddataset     Set to download the dataset, default is to not download. Needs internet.
```
