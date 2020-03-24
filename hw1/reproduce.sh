data_root=$1
output_file=$2
wget -O rnnt_latest.pth https://www.dropbox.com/s/idq7abyzkj6ev90/rnnt_latest.pth?dl=1
sed -i "s#DATA_PATH_TO_BE_CHANGED#${data_root}#g" config/dlhlp/asr_dlhlp_layer5_w_pretrained_rnnt.yaml
sed -i "s#OUTPUT_FILE_TO_BE_CHANGED#${output_file}#g" config/dlhlp/decode_rnnt_beam.yaml
python main.py --config config/dlhlp/decode_rnnt_beam.yaml --test --njobs ${NJOBS}
