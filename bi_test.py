import numpy as np
SEED=987654

DICT = {'01': 0, '07': 1, '04': 2, '05': 3}
label=[]
with open('../IEMOCAP/IEMOCAP_bitest_{}.csv'.format(SEED),'r') as f:
    fr=f.readlines()
for line in fr:
    label.append(DICT[line.replace('\n', '').split('\t')[3]])
audio_results = np.load('test/CNN_mfcc_SERtest_{}.npy'.format(SEED))
audio_results=audio_results.squeeze()
audio_label=np.argmax(audio_results,1)
with open('/program/xumingke/bert/base/{}/test_results.tsv'.format(SEED),'r') as f:
    fr=f.readlines()
trans_results=[]
for line in fr:
    num = line.replace('\n', '').split('\t')
    trans_results.append([float(num[0]), float(num[1]), float(num[2]), float(num[3])])
trans_results=np.array(trans_results)
trans_label=np.argmax(trans_results,1)
weight=[]
for i in range(19):
    weight.append((i+1)*0.05)
for w in weight:
    bi_corr = 0
    bi_class = [0, 0, 0, 0]
    bi_class_corr = [0, 0, 0, 0]
    bi_results = w * trans_results + (1 - w) * audio_results
    bi_label = np.argmax(bi_results, 1)
    for i in range(len(label)):
        if (bi_label[i] == label[i]):
            bi_corr += 1
            bi_class_corr[label[i]] += 1
        bi_class[label[i]] += 1
    UA = 0
    for i in range(4):
        UA += bi_class_corr[i] / bi_class[i]
    UA = UA / 4
    print(str(bi_corr / len(label)) + ',' + str(UA))
