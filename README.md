# CStrCRL
Public code for [CStrCRL](https://ieeexplore.ieee.org/document/10239180).

We are preparing to release the code and relevant files. 

If you have any questions about the code, please email me, and I will respond ASAP. 

# Data
* NTU-RGB D 60/120
* PKUMMD II

# Train
`python main.py pretrain_CStrCRL_ST --config=[myconfig/xxx.yaml] `

# Test

`python main.py linear_evaluation_ST --config=[myconfig/xxx.yaml] `

# Acknowledgement

We sincerely thank the authors for releasing the code of their valuable works. Our code is based on the following repos.
* The code of our framework is heavily based on [HiCLR](https://github.com/JHang2020/HiCLR)
* The code of encoder is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md)
