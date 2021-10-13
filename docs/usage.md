# 1. Inference
## 1.1. Configuration
### 1.1.1. Model text detection

Supported Algorithms:

<details open>
<summary>Text Detection</summary>

| Algorithm      | Paper | Python argument (--det) |
| :---        |    :----:   | ---: |
|- [x] [DBNet](configs/textdet/dbnet/README.md) (AAAI'2020) |https://arxiv.org/pdf/1911.08947|  DB_r18, DB_r50|
|- [x] [Mask R-CNN](configs/textdet/maskrcnn/README.md) (ICCV'2017)|https://arxiv.org/abs/1703.06870|MaskRCNN_CTW, MaskRCNN_IC15, MaskRCNN_IC17|
|- [x] [PANet](configs/textdet/panet/README.md) (ICCV'2019)|https://arxiv.org/abs/1908.06391|PANet_CTW, PANet_IC15|
|- [x] [PSENet](configs/textdet/psenet/README.md) (CVPR'2019)|https://arxiv.org/abs/1903.12473|PS_CTW, PS_IC15|
|- [x] [TextSnake](configs/textdet/textsnake/README.md) (ECCV'2018)|https://arxiv.org/abs/1807.01544|TextSnake|
|- [x] [DRRG](configs/textdet/drrg/README.md) (CVPR'2020)|https://arxiv.org/abs/2003.07493|DRRG|
|- [x] [FCENet](configs/textdet/fcenet/README.md) (CVPR'2021)|https://arxiv.org/abs/2104.10442|FCE_IC15, FCE_CTW_DCNv2|

</details>

**Table 1**: Text detection algorithms, papers and parameters configuration in SDK.

### 1.1.2. Model text recognition

<details open>
<summary>Text Recognition</summary>

| Algorithm      | Paper | Python argument --recog | 
| :---        |    :----:   |---:|
|- [x] [CRNN](configs/textrecog/crnn/README.md) (TPAMI'2016)|https://arxiv.org/abs/1507.05717| CRNN, CRNN_TPS |
|- [x] [NRTR](configs/textrecog/nrtr/README.md) (ICDAR'2019)|https://arxiv.org/abs/1806.00926| NRTR_1/8-1/4, NRTR_1/16-1/8|
|- [x] [RobustScanner](configs/textrecog/robust_scanner/README.md) (ECCV'2020)|https://arxiv.org/abs/2007.07542| RobustScanner |
|- [x] [SAR](configs/textrecog/sar/README.md) (AAAI'2019)|https://arxiv.org/abs/1811.00751| SAR |
|- [x] [SATRN](configs/textrecog/satrn/README.md) (CVPR'2020 Workshop on Text and Documents in the Deep Learning Era)|https://arxiv.org/abs/1910.04396| SATRN, SATRN_sm | 
|- [x] [SegOCR](configs/textrecog/seg/README.md) (Manuscript'2021)|-| SEG |

</details>

**Table 2**: Text recognition algorithms, papers and parameters configuration in SDK.

## 1.2. Inference

```shell
# Activate your conda environment
conda activate gen_ocr

python general_ocr/utils/ocr.py demo/demo_text_ocr.jpg --print-result --imshow --det PANet_IC15 --recog SEG
```
`--det` values and `--recog` values are supplied in **table 1** and **table 2**.

--TODO Result demo

# 2. Training


