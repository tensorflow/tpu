#! /bin/bash -e

source /etc/lsb-release

echo "Verifying build packages are installed..."
sudo sed -Ei "s/# *(deb-src .* main restricted$)/\1/" /etc/apt/sources.list
sudo apt-get update

sudo apt-get build-dep "linux-image-$(uname -r)" -y

echo "Building modules..."

rm -f *.ko
make -f Makefile.kernel_driver_modules

echo "Modules are:"
for module in $(ls -1 *.ko); do
  echo -e "\t$module"
done
echo -e "They will only work on kernel:\n\t$(uname -r)"
