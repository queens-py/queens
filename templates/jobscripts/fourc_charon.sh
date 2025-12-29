#!/bin/bash
# Setup shell environment and start from home dir
echo $HOME
cd $HOME

module list
##########################################
#                                        #
#  Specify the paths                     #
#                                        #
##########################################

RUN_FOUR_C="ON"
EXE={{ executable }}
FOUR_C_BUILD_DIR="$(dirname {{ executable }})"

INPUT={{ input_file }}
FOUR_C_OUTPUT_DIR={{ output_dir }}
OUTPUT_PREFIX="$(basename {{ output_file }})"


##########################################
#                                        #
#  Postprocessing                        #
#                                        #
##########################################
DoPostprocess=$[ ! -z "{{ post_processor|default("") }}" ]
if [ $DoPostprocess ]
then
  RUN_ENSIGHT_FILTER="ON"
else
  RUN_ENSIGHT_FILTER="OFF"
fi

ENSIGHT_OUTPUT_DIR={{ output_dir }}
ENSIGHT_OPTIONS={{ post_options|default("") }}


##########################################
#                                        #
#  RESTART SPECIFICATION                 #
#                                        #
##########################################

RESTART_FROM_STEP=0                 	# specify the restart step here and in .datfile
RESTART_FROM_DIR=""			# same as output
RESTART_FROM_PREFIX="" 		# prefix typically s

#################################################################
# BEGIN ############### DO NOT TOUCH ME #########################
#################################################################

# execute program
source /home/cluster_tools/core/charon_job_core
trap 'EarlyTermination; StageOut' 2 9 15 18
DoChecks
StageIn
RunProgram
wait
StageOut
# show
# END ################## DO NOT TOUCH ME #########################
echo
echo "Job finished with exit code $? at: `date`"
