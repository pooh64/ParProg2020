#! /bin/bash
# run.sh <number of repetitions>
compiler="g++"
flags="-fopenmp -std=gnu++11 -Wall -Wextra -Werror"
src="./src/main.cpp"
build="./build"
exe="$build/task"
tests_dir="./tests"

echo "[CLEAR]"
echo "  rm $build -r"
rm $build -rf

echo "[BUILD]"
echo "  mkdir $build"
mkdir $build
echo "  $compiler $src -o $exe $flags"
$compiler $src -o $exe $flags

if [ ! $? -eq 0 ]; then
  echo "[ERROR] Can't compile $src"
  exit 1
fi

SUCCESS_TESTS=()
FAIL_TESTS=()

for test_dir in $tests_dir/*; do
  test=$(basename $test_dir)
  printf "\n[TEST $test]\n"
  echo "  $exe $test_dir/input.txt $build/$test.txt"
  START=$(date +%s%N)
  $exe $test_dir/input.txt $build/$test.txt
  END=$(date +%s%N)
  DIFF=$((($END - $START)/1000000))
  if [ ! $? -eq 0 ]; then
    echo "[TEST $test] RUNTIME FAIL"
    continue;
  fi
  if cmp -s $build/$test.txt $test_dir/output.txt; then
    echo "[TEST $test] OK ($DIFF ms)"
    RES="OK"
    if [ -n "$1" ]; then
      for ((i=1; i < $1; i++)); do
        START=$(date +%s%N)
        $exe $test_dir/input.txt $build/${test}_$i.txt
        END=$(date +%s%N)
        DIFF=$(($DIFF + ($END - $START)/1000000))
        if ! cmp -s $build/${test}_$i.txt $test_dir/output.txt; then
          RES="FAIL"
          echo "DIFF FAIL: vimdiff $build/${test}_$i.txt $test_dir/output.txt"
          FAIL_TESTS+=($test)
        fi
      done
      echo "[TEST ${test}x$1] $RES ($DIFF ms)"
    fi
    if [[ $RES=="OK" ]]; then
      SUCCESS_TESTS+=($test)
    fi
  else
    echo "[TEST $test] DIFF FAIL($DIFF ms): vimdiff $build/$test.txt $test_dir/output.txt"
    FAIL_TESTS+=($test)
  fi
done
echo "==========="
echo "SUCCESSFUL: ${SUCCESS_TESTS[@]}"
echo "FAIL: ${FAIL_TESTS[@]}"
