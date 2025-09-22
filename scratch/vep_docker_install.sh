docker run --rm -it \
  --user $(id -u):$(id -g) \
  -e TMPDIR=/opt/vep/.vep/tmp \
  -v "$(pwd)/vep_cache:/opt/vep/.vep" \
  -v "$HOME/vep_data:/data" \
  ensemblorg/ensembl-vep \
  bash -lc 'set -euo pipefail
    cd /opt/vep/src/ensembl-vep
    mkdir -p /opt/vep/.vep/tmp
    perl INSTALL.pl -a cf -s homo_sapiens -y GRCh38 --ASSEMBLY GRCh38 --CACHEDIR /opt/vep/.vep'