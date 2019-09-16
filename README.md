# Decision-Tree
Decision Tree implementation from Scratch.

#### Contributors
1. Omkar Thawakar (Research Assistant, IIT Ropar)
2. Alok Jadhav (MS. University of Utah)

Following code build the decision tree using ID3 and GINI index method.
Our code require config file in the following format.
```
{
   'data_file' : PATH_TO_TRAING_DATA (CSV file),
   'data_mappers' : [],
   'data_project_columns' : [COLUMNS_IN_CSV_FILE] ,
   'target_attribute' : 'label'
}
```
We provide sample training data in data folder.
Structure of data folder is 
    data
    ├── .csv          # Training Data
    ├── .csv          # Testing data
    ├── CVfolds           # files for 5 fold cross validation
    │   ├── .cfg          # Load and stress tests
    |   ├── .csv          # Training Data
    │   └── .csv          # Testing data
    └── ...


To Build a Decision Tree using ID3 
```
python id3.py 'data/data.cfg'
```
Output 
```
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS w AND cap-shape EQUALS b AND cap-surface EQUALS y AND cap-color EQUALS w THEN e
==================================================
IF spore-print-color EQUALS k AND gill-size EQUALS b AND gill-color EQUALS p AND stalk-color-below-ring EQUALS g AND cap-color EQUALS g AND cap-surface EQUALS f AND stalk-color-above-ring EQUALS w THEN p
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS k AND cap-shape EQUALS f AND cap-color EQUALS e THEN e
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS w AND cap-shape EQUALS x AND cap-color EQUALS y THEN e
==================================================

==================================================
Error on train.csv is  ::: 0.0 %
Error on test.csv is  ::: 11.711711711711711 %
Maximum Depth of Tree is :  17
```

To Build a Decision Tree using GINI 
```
python gini.py 'data/data.cfg'
```
Output 
```
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS w AND cap-shape EQUALS b AND cap-surface EQUALS y AND cap-color EQUALS w THEN e
==================================================
IF spore-print-color EQUALS k AND gill-size EQUALS b AND gill-color EQUALS p AND stalk-color-below-ring EQUALS g AND cap-color EQUALS g AND cap-surface EQUALS f AND stalk-color-above-ring EQUALS w THEN p
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS k AND cap-shape EQUALS f AND cap-color EQUALS e THEN e
==================================================
IF spore-print-color EQUALS n AND gill-size EQUALS b AND gill-color EQUALS w AND cap-shape EQUALS x AND cap-color EQUALS y THEN e
==================================================

==================================================
Depth of Decision tree is  :  8
Error on train.csv is  ::: 0.0 %
Error on test.csv is  ::: 12.612612612612613 %
```

For 5 fold cross validation with limiting depth

```
python limiting_depth.py 'data/data.cfg'
```
Output
```
==================================================
Depth :  1
Cross Validation Accuracy on fold2.csv ::: 0.7518796992481203
==================================================
Depth :  2
Cross Validation Accuracy on fold2.csv ::: 3.007518796992481
==================================================
Depth :  3
Cross Validation Accuracy on fold2.csv ::: 10.902255639097744
==================================================
Depth :  4
Cross Validation Accuracy on fold2.csv ::: 54.51127819548872
==================================================
Depth :  5
Cross Validation Accuracy on fold2.csv ::: 76.69172932330827
==================================================
Depth :  10
Cross Validation Accuracy on fold2.csv ::: 85.71428571428571
==================================================
Depth :  15
Cross Validation Accuracy on fold2.csv ::: 85.71428571428571
##################################################
Max Accuracy is 85.71428571428571 for fold 10 .
##################################################
..
...
....
.....
##################################################
Max Accuracy is 85.71428571428571 for fold 10 .
##################################################

```
