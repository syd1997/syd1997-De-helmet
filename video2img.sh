#!/usr/bin/env bash
{
  remove_tailing_slash() {
    local path=${1-data//xm0901///}
    path=$(tr -s / <<<"$path")
    path=${path%/}
    echo "$path"
  }
  video2img() {
    local usage="${FUNCNAME[0]} src dst [--poolsize=1]"
    eval "$parse_set_pos_macro"
    local src=${1:-"$prefix"_video} dst=${2:-"$prefix"} src_file dst_file
    src=$(remove_tailing_slash "$src")
    while IFS= read -r -d '' src_file; do
      dst_file=${src_file#"$src"}
      dst_file=$dst${dst_file%.*}
      mkdir -p "$dst_file"
      echo "$src_file" "$dst_file/%06d.jpg"
      ffmpeg -loglevel error -hide_banner -stats -nostdin -i "$src_file" -qscale:v 2 "$dst_file/%06d.jpg" &
      poolwait ${opt[poolsize]:-1}
    done < <(find -L "$src" -type f -print0)
    wait
  }
  "$@"
  wait
  date "+%H:%M:%S %m%d"
  exit
}