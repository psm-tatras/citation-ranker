curl -4 ifconfig.me

ssh -i "/Work/AWS Credentials/PSMAWSEC2.pem" ubuntu@ec2-65-2-10-22.ap-south-1.compute.amazonaws.com
scp -i "/Work/AWS Credentials/PSMAWSEC2.pem" setup_ec2.sh ubuntu@ec2-65-2-10-22.ap-south-1.compute.amazonaws.com:/home/ubuntu/qwen-ner-finetuning
scp -i "/Work/AWS Credentials/PSMAWSEC2.pem" ubuntu@ec2-65-2-10-22.ap-south-1.compute.amazonaws.com:/home/ubuntu/qwen-ner-finetuning/train_qlora.py "/Work/Experiments/citation-processor-ranker/training/aws_setup
/train_qlora.py"

scp -i "/Work/AWS Credentials/PSMAWSEC2.pem" -r ubuntu@ec2-65-2-10-22.ap-south-1.compute.amazonaws.com:/opt/dlami/nvme/qwen-ner-adapter "/Work/Experiments/citation-processor-ranker/models/ner/qwen-10k/"


python3 prepare_highlights_data.py --csv "/Work/Experiments/citation-processor-ranker/data/mixsub-fire/MixSub-SciHigh_train_FIRE.csv" --output "/Work/Experiments/citation-processor-ranker/data/mixsub-fire/processed/slm_train/summary/fire_mixsub_10k_train.jsonl"

python3 prepare_highlights_data.py --csv "/Work/Experiments/citation-processor-ranker/data/mixsub-fire/MixSub-SciHigh_val_FIRE.csv" --output "/Work/Experiments/citation-processor-ranker/data/mixsub-fire/processed/slm_train/summary/fire_mixsub_10k_test.jsonl"

python3 train_highlights_qlora.py --train-file "summary/fire_mixsub_10k_train.jsonl" --val-file "summary/fire_mixsub_10k_test.jsonl" --num-val-samples 20 --output-dir /opt/dlami/nvme/qwen-summary-adapter