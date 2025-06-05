# har_rgbe

## Training Script

`train.py` now supports loading a pretrained model before training. Use
the `--pretrained` argument to specify the weight file:

```bash
python train.py --config path/to/config.yaml \
                --model path/to/save_best.pth \
                --log path/to/log.txt \
                --pretrained path/to/pretrained_weights.pth
```
