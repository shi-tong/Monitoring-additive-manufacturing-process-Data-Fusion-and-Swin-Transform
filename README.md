# Monitoring-additive-manufacturing-process-Data-Fusion-and-Swin-Transform

This repository contains the Python code for the paper "Data fusion and Swin Transform architecture-based model for monitoring geometric morphology of blue laser directed energy deposited Al-Mg alloy"

&#x20;

WARNING: These codes are written only for the purpose of demonstration and verification. While the correctness has been carefully checked, the quality such as standardability, clarity, generality, and efficiency has not been well considered.

&#x20;

In the present study, to enhance the accuracy of monitoring the geometry of the deposited layer, we proposed a dual-layer hybrid model with two fusion techniques, i.e., a re-parameterization-based real-time end-to-end fusion and an attention-based wavelet transform fusion.&#x20;

The a re-parameterization-based real-time end-to-end fusion model is as follows:&#x20;

1.The gray fusion of molten pool image and infrared image is realized by the model. (.../dual-layer hybrid  model/ LDRepFM fusion / modules/model.py).&#x20;

The attention-based wavelet transform fusion model is as follows:

2.The wavelet transform model based on the attention mechanism includes three inputs: the grayscale texture image obtained through the fusion of the LDRepFM model, the melt pool geometry image, and the infrared temperature image.  (.../dual-layer hybrid  model/ attention-based wavelet transform fusion.py).

&#x20;

Additionally, a morphology prediction model based on the focused parallel activation module (FPAM)-Swin Transformer was proposed as a monitoring algorithm. A performance comparison was made with the Swin Transformer, InceptionResNetV2, and DenseNet models to demonstrate the superior accuracy of our proposed method.&#x20;

1.We integrated a PAM and focused linear attention module into Swin Transformer model to form the FPAM-Swin Transformer model.(.../monitoring models/ FPAMSwinTransformer.py).&#x20;

2.The Swin Transformer model.(.../monitoring models/ Swin-Transformer-main/model/swin_transformer.py).&#x20;

3.The InceptionResNetV2 model.(.../monitoring models/ InceptionResNetV2.py).&#x20;

4.The DenseNet model.(.../monitoring models/ DenseNet.py).
