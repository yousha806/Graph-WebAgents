import os
import sys
from dotenv import load_dotenv

print('cwd=', os.getcwd())
print('python version=', sys.version)
load_dotenv()

token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
print('HUGGINGFACE_TOKEN present:', bool(token))
if not token:
    print('No token found in environment or .env')
    sys.exit(0)

try:
    from huggingface_hub import HfApi
except Exception as e:
    print('Error importing huggingface_hub:', e)
    sys.exit(1)

api = HfApi()
try:
    info = api.whoami(token=token)
    print('whoami success:')
    print(info)
except Exception as e:
    print('whoami failed:', e)
    sys.exit(2)
