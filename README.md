conda create -n bench-env python=3.10 -y
conda activate bench-env
conda install pytorch cpuonly -c pytorch -y
pip install -r requirements.txt
