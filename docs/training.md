# Training

## Training on a Single Machine


You can use `tools/train.py` to train a model in a single machine with one or more GPUs.

Here is the full usage of the script:

```shell
python tools/train.py ${CONFIG_FILE} [ARGS]
```


| ARGS      | Type                  |  Description                                                 |
| -------------- | --------------------- |  ----------------------------------------------------------- |
| `--work-dir`          | str                   |  The target folder to save logs and checkpoints. Defaults to `./work_dirs`. |
| `--load-from`   | str                   |  The checkpoint file to load from. |
| `--resume-from`        | bool |  The checkpoint file to resume the training from.|
| `--no-validate` | bool |  Disable checkpoint evaluation during training. Defaults to `False`. |
| `--gpus`       | int                   |  Numbers of gpus to use. Only applicable to non-distributed training. |
| `--gpu-ids`       | int*N                   | A list of GPU ids to use. Only applicable to non-distributed training. |
| `--seed`      | int                   |  Random seed. |
| `--deterministic`       | bool                   |  Whether to set deterministic options for CUDNN backend. |
| `--cfg-options`       | str                   |          Override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the config file. If the value to be overwritten is a list, it should be of the form of either key="[a,b]" or key=a,b. The argument also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks are necessary and that no white space is allowed.|
| `--launcher`       | 'none', 'pytorch', 'slurm', 'mpi' |  Options for job launcher. |
| `--local_rank`       | int                   |Used for distributed training.|
| `--mc-config`       | str                   |Memory cache config for image loading speed-up during training.|
