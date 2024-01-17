#!/bin/bash

if [ $# -ne 1 ]
then
    echo "$0 <pname>"
    exit 1
fi

ps_num=`ps -efww | grep $1 | grep -v grep | grep -v $0 | wc -l`
if [ $ps_num -gt 0 ]
then
    ps -efww | grep $1 | grep -v grep | grep -v $0 | awk '{print int($3)}' | xargs kill -9
    ps -efww | grep $1 | grep -v grep | grep -v $0 | awk '{print int($2)}' | xargs kill -9
fi
