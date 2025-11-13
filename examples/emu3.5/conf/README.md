# Quick Start

vLLM implementation of https://github.com/baaivision/Emu3.5

## Prepare

```
# install FlagScale(vLLM)
git clone https://github.com/flagos-ai/FlagScale.git
cd FlgScale
python tools/patch/unpatch.py --backend vllm



# git clone `src/` from Emu3.5
git clone --no-checkout https://github.com/baaivision/Emu3.5.git tmp_repo
cd tmp_repo
git sparse-checkout init --cone
git sparse-checkout set src assets
git checkout main
mv src assets ../
cd ..
rm -rf tmp_repo

pip install flash_attn==2.8.3 --no-build-isolation
```








