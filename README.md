# Inference-Time Policy Steering (ITPS)

Maze2D benchmark of various sampling methods with sketch input from the paper [Inference-Time Policy Steering through Human Interactions](https://yanweiw.github.io/itps/).

## Installation 
Clone this repo
```
git clone git@github.com:yanweiw/itps.git
cd itps
```
Create a virtual environment with Python 3.10
```
conda create -y -n itps python=3.10
conda activate itps
```
Install ITPS
```
pip install -e .
```
Download the pre-trained weights for [Action Chunking Transformers](https://drive.google.com/file/d/1kKt__yQpXOzgAGFvfGpBWdtWX_QxWsVK/view?usp=sharing) and [Diffusion Policy](https://drive.google.com/file/d/1efez47zfkXl7HgGDSzW-tagdcPj1p8z2/view?usp=sharing) and put them in the `itps/itps` folder. 

## Visualize pre-trained policies. 

Run ACT or DP unconditionally to explore motion manifolds learned by these pre-trained policies.
```
python interact_maze2d.py -p [act, dp] -u
```
|Multimodal predictions of DP|
|---------------------------|
|![](media/dp_manifold.gif)|


## Bias sampling with sketch interaction. 

`-ph` - Post-Hoc Ranking
`-op` - Output Perturbation
`-bi` - Biased Initialization
`-gd` - Guided Diffusion
`-ss` - Stochastic Sampling
```
python interact_maze2d.py -p [act, dp] [-ph, -bi, -gd, -ss]
```
|Post-Hoc Ranking Example|
|---------------------------|
|![](media/pr_example.gif)|
Draw by clicking and dragging the mouse. Re-initialize the agent (red) position by moving the mouse close to it without clicking. 

## Visualize sampling dynamics.

Run DP with BI, GD or SS with `-v` option.
```
python interact_maze2d.py -p [act, dp] [-bi, -gd, -ss] -v
```
| Stochastic Sampling Example|
|---------------------------|
|![](media/ss_dynamics.gif)|

## Benchmark methods.
Save sketches into a file `exp00.json` and use them across methods.
```
python interact_maze2d.py -p [act, dp] -s exp00.json
```
Visualize saved sketches by loading the saved file, press the key `n` for next. 
```
python interact_maze2d.py -p [act, dp] [-ph, -op, -bi, -gd, -ss] -l exp00.json
```
Save experiments into `exp00_dp_gd.json`
```
python interact_maze2d.py -p dp -gd -l exp00.json -s .json
```
Replay experiments.
```
python interact_maze2d.py -l exp00_dp_gd.json
```

## Acknowledgement

Part of the codebase is modified from [LeRobot](https://github.com/huggingface/lerobot).
