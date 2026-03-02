Creating EC2 instance:
* Ubuntu 24.04 instance 
* SSH from my IP
* 200 GB
* mind2web-key.pem

Connecting to instance:
* ssh -i mind2web-key.pem -L 8888:localhost:8888 ubuntu@<public-ip>
* Delete after use because of memory

Instance setup:
* <sudo apt-get update>
* <sudo apt-get upgrade>
* <sudo apt install -y nvidia-driver-535-server>
* <sudo reboot> -- This reboots the instance, reconnect after waiting for 10s
* Miniconda install:
  * <wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh>
  * <bash ~/miniconda.sh -b -p ~/miniconda3>
  * <eval "$(~/miniconda3/bin/conda shell.bash hook)">
* Create conda env:
  * <conda create -n mind2web python=3.10> This will prompt to accept Tos, put <a> both times
  * <conda activate mind2web>
* Repo clone
  * <git clone https://github.com/yousha806/Graph-WebAgents.git>
  * <git checkout mru1> -- Or any other branch in place of mru1 with the code
  * <cd Graph-WebAgents>
* Install requirements
  * <pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121>
  * <pip install -r requirements.txt>
  * <playwright install chromium>
  * <sudo env PATH="$PATH" playwright install-deps chromium> -- Separate chromium setup
  * <python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"> -- Verify GPU


