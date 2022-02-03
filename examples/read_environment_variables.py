import os

from decouple import config

# You can load your own environment variables
# Defined either in the terminal or in `.env`
# Both `config('NAME')` and `os.environ['NAME']` will give the same result: the value as a string
#   The difference between them are that:
#   1. The former (config) is more robust and can be typecased
#   2. You can specify default values if the value is not set
#   just to name a few
BATCH_SIZE_str = config("BATCH_SIZE")
print(BATCH_SIZE_str, type(BATCH_SIZE_str))
BATCH_SIZE = config("BATCH_SIZE", cast=int)

NUM_GPUS = config("PYTHON_ENV_NVIDIA_VISIBLE_DEVICES", cast=int, default=0)
print(NUM_GPUS, type(NUM_GPUS))

# See all environment variables available in the Docker container
print(os.environ)
