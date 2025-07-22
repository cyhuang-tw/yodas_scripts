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
cd /ocean/projects/cis210027p/chuang14/espnet/tools
. ./activate_python.sh
cd /ocean/projects/cis210027p/chuang14/yodas_scripts

nj=1000
_logdir=logdir/ctc_seg
key_file=data_reseg/json_files.txt

mkdir -p ${_logdir}

_nj=${nj}

# _nj=$(min "${nj}" "$(wc -l ${key_file})")
split_scps=""
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/data.${n}"
done
utils/split_scp.pl "${key_file}" ${split_scps}

./slurm.pl --max-jobs-run 100 --gpu 1 --time 2:00:00 JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
    python ctc_seg.py --file_list ${_logdir}/data.JOB --root_dir /ocean/projects/cis210027p/takamich/archive_org/audio
