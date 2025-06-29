git clone https://github.com/openai/simple-evals
cd simple-evals
git checkout 6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58
sed -i 's/\r//g' ../simple_evals_for_vllm.patch
git apply ../simple_evals_for_vllm.patch
cd ..
cp -r simple-evals/* ./
rm -rf simple-evals/
sed -i '66a\                    timeout=7200,' ./sampler/chat_completion_sampler.py
mkdir results

simple_evals_dataset_url='https://openaipublic.blob.core.windows.net/simple-evals'

wget ${simple_evals_dataset_url}/mmlu.csv -P ./dataset/mmlu

wget ${simple_evals_dataset_url}/gpqa_diamond.csv -P ./dataset/gpqa

wget ${simple_evals_dataset_url}/mgsm_bn.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_de.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_en.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_es.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_fr.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_ja.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_ru.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_sw.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_te.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_th.tsv -P ./dataset/mgsm
wget ${simple_evals_dataset_url}/mgsm_zh.tsv -P ./dataset/mgsm

wget ${simple_evals_dataset_url}/drop_v0_train.jsonl.gz -P ./dataset/drop
wget ${simple_evals_dataset_url}/drop_v0_dev.jsonl.gz -P ./dataset/drop

git clone https://github.com/openai/human-eval
pip install -e human-eval
pip install openai
pip install anthropic
pip install httpx==0.27.2
