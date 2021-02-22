# Bimodal-ER

This repository is based on Bert (https://github.com/google-research/bert) and IEMOCAP data set (https://sail.usc.edu/iemocap/iemocap_publication.htm)

1. Train a text classification model based on Bert:
  (1) Download a pre-trained Bert model like 'uncased_L-12_H-768_A-12.zip'(Bert-Base)
  (2) fine-tune the pre-trained model on the IEMOCAP data set with 'run_classifier.py'
2. Train a speech classification model with Head Fusion (https://github.com/lessonxmk/head_fusion) or Area Attention (https://github.com/lessonxmk/Optimized_attention_for_SER)
3. Run two models above on the test data set to obtain the numpy file of the model's predicted probability
4. Run 'Bi_test.py' to obtain the final accuracy of the model combined with the two models.
