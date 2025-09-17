# With pip:
%pip install facenet-pytorch

# or clone this repo, removing the '-' to allow python imports:
%git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

# or use a docker container (see https://github.com/timesler/docker-jupyter-dl-gpu):
%docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython