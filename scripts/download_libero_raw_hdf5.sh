#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/Video-as-action-generation-source/mimic-video/data/libero_raw}
BASE_URL=${LIBERO_HF_BASE_URL:-https://hf-mirror.com/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main}

mkdir -p "${OUT_DIR}"

download_one() {
  local rel=$1
  local expected_size=$2
  local dst="${OUT_DIR}/${rel}"
  mkdir -p "$(dirname "${dst}")"

  if [[ -f "${dst}" ]]; then
    local got
    got=$(stat -c '%s' "${dst}")
    if [[ "${got}" == "${expected_size}" ]]; then
      echo "skip complete: ${rel} (${got} bytes)"
      return 0
    fi
    echo "resume: ${rel} (${got}/${expected_size} bytes)"
  else
    echo "download: ${rel} (${expected_size} bytes)"
  fi

  local attempt=1
  until wget -c \
      --tries=20 \
      --timeout=30 \
      --read-timeout=120 \
      --retry-connrefused \
      --waitretry=5 \
      --progress=dot:giga \
      -O "${dst}" \
      "${BASE_URL}/${rel}"; do
    if [[ "${attempt}" -ge "${LIBERO_WGET_OUTER_TRIES:-8}" ]]; then
      echo "ERROR: failed after ${attempt} attempts: ${rel}" >&2
      exit 1
    fi
    attempt=$((attempt + 1))
    echo "retry ${attempt}/${LIBERO_WGET_OUTER_TRIES:-8}: ${rel}"
    sleep "${LIBERO_WGET_OUTER_SLEEP:-10}"
  done

  local final_size
  final_size=$(stat -c '%s' "${dst}")
  if [[ "${final_size}" != "${expected_size}" ]]; then
    echo "ERROR: size mismatch for ${rel}: got ${final_size}, expected ${expected_size}" >&2
    exit 1
  fi
}

download_one "libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5" 702223367
download_one "libero_goal/open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5" 1017886342
download_one "libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5" 762855139
download_one "libero_goal/put_the_bowl_on_the_plate_demo.hdf5" 468246288
download_one "libero_goal/put_the_bowl_on_the_stove_demo.hdf5" 509123820
download_one "libero_goal/put_the_bowl_on_top_of_the_cabinet_demo.hdf5" 510414285
download_one "libero_goal/put_the_cream_cheese_in_the_bowl_demo.hdf5" 535715512
download_one "libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5" 878958730
download_one "libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5" 540179470
download_one "libero_goal/turn_on_the_stove_demo.hdf5" 447509922

download_one "libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5" 508779600
download_one "libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5" 589630943
download_one "libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5" 748271833
download_one "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5" 632345992
download_one "libero_spatial/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5" 597677074
download_one "libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5" 671583385
download_one "libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5" 507189857
download_one "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5" 581087722
download_one "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5" 711715594
download_one "libero_spatial/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5" 688768764

download_one "libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5" 780145352
download_one "libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5" 734212272
download_one "libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5" 785532626
download_one "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5" 797322420
download_one "libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5" 718919781
download_one "libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5" 804674488
download_one "libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5" 727315202
download_one "libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5" 696180446
download_one "libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5" 664188039
download_one "libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5" 735593408
