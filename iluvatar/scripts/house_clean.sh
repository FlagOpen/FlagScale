#! /usr/bin/env bash


if [ -d ~/.share/Downloads/corex/bi100-driver.r311.ub2004.`uname -r` -a \
     -f ~/workspace/utils/house_clean_gpus_iluvatar.sh ]; then true \
 && env kernel_module_dir=~/.share/Downloads/corex/bi100-driver.r311.ub2004.`uname -r` \
    bash ~/workspace/utils/house_clean_gpus_iluvatar.sh \
 && true; \
fi
