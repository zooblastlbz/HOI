set -ex

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


num_slots=1
sed -i "s/slots=[0-9]\+\$/slots=$num_slots/g" /etc/mpi/hostfile

mpirun -x PATH -hostfile /etc/mpi/hostfile python /ytech_m2v5_hdd/workspace/kling_mm/dingyue08/spatial-r1/HOI/HOI/pipeline/mpi_rewrite.py