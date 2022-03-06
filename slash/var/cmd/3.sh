#!/bin/bash
m="start"
while [ "$m" = "start" ] || [ "$m" = "r" ] || [ "$m" = "R" ]; do
  [ "$m" != "start" ] && builtin echo -e "\n\n\n\n\e[31mRECONNECTING, PLEASE WAIT...\e[0m\n\n\n\n"
  ssh "cl6418@greene.hpc.nyu.edu" -l PleaseAskMe -p 22 -Y -o ForwardX11Trusted=yes -o ForwardX11=yes  -C -o Compression=yes  -Z 
  i0=0
  s0=""
  while [ "$i0" -lt "${COLUMNS:-80}" ]; do
    s0="q$s0"
    i0=$[$i0+1]
  done
  builtin echo -e "\n\e[1;30m\e(0$s0\e(B\e[0m
  \n\e[31mSession stopped\e[0m
    - Press \e[35m<return>\e[0m to exit tab
    - Press \e[35mr\e[0m to restart session
    - Press \e[35ms\e[0m to save terminal output to file\n"
  m="s"
  while [ "$m" = "s" ] || [ "$m" = "S" ]; do
    read -s -n 1 m
    [ "$m" = "S" ] || [ "$m" = "s" ] && MobaSendMsg MobaXterm "SessionSave $TERMNUM"
  done
done
