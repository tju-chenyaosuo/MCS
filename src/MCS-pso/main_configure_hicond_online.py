import os
from main_configure_approach import MainProcessConf, CsmithConfigurationConf

particle_number = 10

tmp_test_program_dir = 'test_conf_programs' + os.sep

c = 2

if_reset = False
limit_velocity = True
single_limitations = [-5, 5]
group_limitations = [-3, 3]
velocity_limitations = []

for _ in range(len(CsmithConfigurationConf.single_keys)):
    velocity_limitations.append(single_limitations[:])
for _ in range(len(CsmithConfigurationConf.single_keys), 71):
    velocity_limitations.append(group_limitations[:])

assert len(velocity_limitations) == 71

