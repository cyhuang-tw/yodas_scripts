while IFS= read -r line; do
    for x in /scratch/bbjs/shared/corpora/yodas2/data/$line/audio/*.tar.gz; do
        echo $x
        tar xzf $x -C $(dirname $x)
        for y in $(dirname $x)/*.wav; do
            flac --force --silent --delete-input-file $y
        done
    done
done < $1
