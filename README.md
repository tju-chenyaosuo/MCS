# MCS
Related Zenodo homepage: https://doi.org/10.5281/zenodo.6521931

We have updated our docker image (```anonymousmcs/mcs_image:v1```) on dockerhub for researchers to run all experiments in the paper.
For more details, please refer to ```/data/code/readme.txt``` in the image.

It cantains two major parts: experiment data(in `data`) and source code(in `src`).

- [MCS](#mcs)
  - [Experiment Data](#experiment-data)
    - [Comparison between MCS and Swarm/HiCOND](#comparison-between-mcs-and-swarmhicond)
    - [Comparison between MCS and MCS-pso](#comparison-between-mcs-and-mcs-pso)
    - [Comparison between MCS and Simple-ml](#comparison-between-mcs-and-simple-ml)
    - [Negative Effect Investigation](#negative-effect-investigation)
    - [Comparison between Different MCS Configurations](#comparison-between-different-mcs-configurations)
    - [Comparison between MCS and its' variants](#comparison-between-mcs-and-its-variants)
  - [How to use MCS](#how-to-use-mcs)
    - [Required Environment](#required-environment)
    - [Configuration](#configuration)
    - [How to run MCS](#how-to-run-mcs)
    - [How to reproduce the experimental results](#how-to-reproduce-the-experimental-results)
  - [Reference](#reference)


## Experiment Data

All of our experiment data is in `data`.

### Comparison between MCS and Swarm/HiCOND

`data/baselines` contains data about **MCS** and compared baselines (i.e. **Swarm** and **HiCOND**), each folder contains the data for the corresponding compiler subject. In each folder, there are two files and two folders:

+ `info.txt`: which contains bug detection results by MCS and compared baselines. 

+ `program_seeds.txt`: which contains the seed to generate each bug-triggering program in info.txt

+ `MCS-conf/`: which contains all the test configurations of MCS that generate bug-triggering programs.

+ `Swarm-conf/`: which contains all the test configurations of Swarm that generate bug-triggering programs.

+ Note: specifically, program identifier is the combination of iteration number (i.e. generation number) and program number in its iteration. 
Furthermore, the name of configuration file in `MCS-conf/` and `Swarm-conf/` is the combination of "config" and its iteration number.
The corresponding configuration file can be found through the number before "-" in a program identifier.

### Comparison between MCS and MCS-pso

`data/MCS-pso` contains data about the comparison between **MCS** and **MCS-pso**

Each compiler version folder contains `info.txt`, `bug-triggering-conf/` and `program_seeds.txt`, and all of them follow the format we mentioned in `data/baselines`.

### Comparison between MCS and Simple-ml

`data/simple-ml` contains data about the comparison with **MCS** and a **simpler machine learning technique**.

We used XGBoost algorithm to construct a regression model to predict a score for each coming random constructed configuration. The overall process is as follows:
  1. Randomly construct 100 test configurations.
  2. Select the configuration that has the biggest predicted score under the built model.
  3. Use the selected configuration to generate 100 test programs to perform compiler testing and evaluate this configuration using our proposed reward function.
  4. Every 10 iterations, use the collected data(i.e., configuration vectors and their rewards) to retrain the XGBoost model.
  5. Return to step 1, until the total run time exceeds 10 days.

Note that all other configurations are as same as MCS, and we built XGBoost model using its' default parameter settings.
    
After 10 days of testing, simple-ml found 4 and 8 bugs on GCC-4.5.0 and LLVM-2.8.0 respectively, while MCS found 17 and 13 bugs respectively. The results confirm the significant superiority of RL.

Each compiler version folder contains `info.txt`, `bug-triggering-conf/` and `program_seeds.txt`, and all of them follow the format we mentioned in `data/baselines`.

### Negative Effect Investigation

`data/negative` contains the data of **option's negative effect investigation**, it has two files:
  + `conf-time.txt`: which contains the number of timeout programs (those running for a long time) generated under each configuration.
    
  + `percentage-timeout_num.txt`: which contains the data that is used to draw Figure 3 in the paper.

### Comparison between Different MCS Configurations

`data/param` contains the data of MCS under **different configurations**. The folders in `data/param` represent the studied parameter name and compiler name. Each compiler version folder contains:

+ four folders: i.e. 5, 10, 15, 20, which is the value of each studied parameter in our experiment. Each of them contains `program_seeds.txt` and `bug-triggering-conf/`, they follow the format we mentioned before (in `data/baselines`).

+ one file: info.txt, following the format we mentioned in `data/baselines`.

### Comparison between MCS and its' variants

`data/variant` contains the data about the comparison of **MCS** and its variants (i.e. **MCS-** and **MCS+**).
The folders in `data/variant` represent the corresponding variants or compiler versions. 
The files inside are `info.txt`, `bug-triggering-conf/` and `program_seeds.txt`, and all of them follow the format we mentioned in `data/baselines`.

## How to use MCS

Note that all code files mentioned in this section can be found in `src`.

### Required Environment

First, some python packages need to be installed:
+ python - 3.7.11
+ pytorch - 1.8.1
+ numpy - 1.19.2
+ scipy - 1.6.2
+ pandas - 1.3.1
+ py-xgboost - 1.5.0

Second, the test program generator is required. To obtain program features, we use a variant of Csmith[1] modified by Chen et al. in HiCOND[2]. To the ease of usage, we redirect the output of the program features, it can be redirected to a file specified by "--record-file". 
This Csmith will be found at "csmith_record_2.3.0_t1.tar.bz2".
To successfully build Csmith, please follow the instructions in https://github.com/csmith-project/csmith

### Configuration

These configuration files are similar in all the folders in `src`:

+ `main_configure_approach.py`: the configuration file of MCS, and the following configuration needs to be set, in order to run all the experiments in the "How to reproduce the experimental results" section:

  + GCCTestConf.gcc: which specifies gcc path.
  
  + GCCTestConf.csmith_lib: which specifies Csmith library path.
  
  + LLVMTestConf.llvm_bin_path: which specifies llvm path.
  
  + LLVMTestConf.csmith_lib: which specifies Csmith library path.
  
  + LLVMTestConf.llvm_ub_clang: which specifies clang to detect undefined behavior (suggest higher version).
    Make sure "-fsanitize=undefined,address" is available in this clang.
  
  + CsmithConfigurationConf.csmith: which specifies Csmith path.
  
  + MainProcessConf.test_compiler_type: which specifies compiler type to be tested.

+ `main_configure_baseline.py`: the configuration file of baselines, needs to be modified to run "baselines".
 
  + baseline_type: which specifies the baseline type, i.e. Swarm or HiCOND.

### How to run MCS
Make sure all settings are specified.

Then run  ```cd MCS; python run-approach.py```

### How to reproduce the experimental results

There are several experiments in this paper:
Necessary configuration should be specified following the instruction in [Configuration](#configuration) section above.

+ baselines: run ```cd MCS; python run-baselines.py```

+ MCS: run ```cd MCS; python run-approach.py```

+ MCS-: run ```cd MCS-; python run-approach.py```

+ MCS+: run ```cd MCS+; python run-approach.py```

+ parameter: run ```cd param/{name}/{value}; python run-approach.py```, 
  in ```param/{name}/{value}```, ```{name}``` means parameter name, i.e. "ch" and "m"; ```{value}``` means parameter value, we pre-configured 5, 10, 15, 20 for each parameter.

+ option's negative effect investigation: run ```cd negative; python run-research.py```

+ MCS-pso: ```cd MCS-pso; python run-hicond-online.py```

+ simple-ml: ```cd simple-ml; python run-simple-ml.py```

***

## Reference

[1] Xuejun Yang, Yang Chen, Eric Eide, and John Regehr. 2011. Finding and under-standing bugs in C compilers. InProceedings of the 32nd ACM SIGPLAN conference on Programming language design and implementation. 283-294.

[2] Junjie Chen, Guancheng Wang, Dan Hao, Yingfei Xiong, Hongyu Zhang, and LuZhang. 2019. History-Guided Configuration Diversification for Compiler Test-Program Generation. In34th IEEE/ACM International Conference on Automated Software Engineering. 305-316.
