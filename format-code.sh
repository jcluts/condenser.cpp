for f in *.cpp *.h *.hpp tools/cli/*.cpp tools/common/*.hpp tools/cli/*.h tools/server/*.cpp; do
  [[ "$f" == vocab* ]] && continue
  echo "formatting '$f'"
  # if [ "$f" != "stable-diffusion.h" ]; then
  #   clang-tidy -fix -p build_linux/ "$f"
  # fi
  clang-format -style=file -i "$f"
done