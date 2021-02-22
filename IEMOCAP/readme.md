The csv files are generate based on IEMOCAP data set with 5 different random seeds.


Among them, 

'train' means the train data set,

'dev' means the dev data set for bert fune-tine,

'test' means the test data set,

'bitest' means the test data set for bimodal test.


About 'bitest' files：

For SER, the 'scripted' part in IEMOCAP data set will decrease the accuracy, so this part will be dropped when training.

But when fune-tining the bert, it's not necessary to drop this part.

So we retain this part in the train/dev/test csv files and only drop it when training SER model.

It causes the difference between bert's and SER's test data set, so we additionally generated the bitest files to ensure the final test files are the same.
