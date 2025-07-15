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

. ./path.sh
. ./cmd.sh

nj=256
_logdir=untar/logdir
key_file=untar/partitions

mkdir -p ${_logdir}

_nj=$(min "${nj}" "$(<${key_file} wc -l)")
split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/partitions.${n}"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}

./slurm.pl JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
    ./untar_file.sh ${_logdir}/partitions.JOB
