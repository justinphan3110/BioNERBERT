import os

os.system('pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html')
os.system('pip install --upgrade clu')
os.system('git clone https://github.com/google/flax.git')
os.system('pip install --user -e flax')
os.system('pip install -r requirements.txt')