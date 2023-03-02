import os

os.system('pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html')
os.system('pip install --upgrade clu')
os.system('pip install --upgrade git+https://github.com/google/flax.git')
os.system('pip install -r requirements.txt')