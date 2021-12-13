### 1. To run CNN training and model pruning

#### Install Numpy package on Linux machine

~~~bash
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
~~~

#### Train and prune the network

~~~bash
cd sw
mkdir models # create dir to save model checkpoints in different epochs
python3 model_training.py
~~~



### 2. To synthesis the HLS hardware code

#### Require vitis_hls installed

~~~bash
cd hw
vitis_hls run_hls.tcl
# to view the report, run `less proj_mulcol/solution1/syn/report/matmul_csynth.rpt`
~~~

To run the second version of systolic array, cover `mulcol.cpp` with `mulcol.cpp.bk`, then redo it.
