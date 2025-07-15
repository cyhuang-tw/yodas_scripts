min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

# module load anaconda3_gpu/23.9.0
# source activate yodas

nj=8192
_logdir=logdir/lid
key_file=data/all.jsonl

mkdir -p ${_logdir}

_nj=${nj}

# _nj=$(min "${nj}" "$(wc -l ${key_file})")
# split_scps=""
# for n in $(seq "${_nj}"); do
#     split_scps+=" ${_logdir}/data.${n}"
# done
# utils/split_scp.pl "${key_file}" ${split_scps}

# ./slurm.pl --max-jobs-run 100 --gpu 1 --time 2:00:00 JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
#     python lid.py --in_file ${_logdir}/data.JOB --out_file ${_logdir}/lid.JOB

cat ${_logdir}/lid.* > data/lid.jsonl
