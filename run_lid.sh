_nj=8000
_logdir=logdir/lid_reseg
key_file=data_reseg/all.jsonl

mkdir -p ${_logdir}

# split_scps=""
# for n in $(seq "${_nj}"); do
#     split_scps+=" ${_logdir}/data.${n}"
# done
# utils/split_scp.pl "${key_file}" ${split_scps}

# ./slurm.pl --max-jobs-run 400 --gpu 1 --time 1:00:00 JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
#     python lid.py --in_file ${_logdir}/data.JOB --out_file ${_logdir}/lid.JOB

cat ${_logdir}/lid.* > data_reseg/lid.jsonl
