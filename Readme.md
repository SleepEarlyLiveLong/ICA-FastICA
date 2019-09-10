# ICA-FastICA

This is a realization of Independent Component Analysis (ICA) and FastICA (Code + Description). Here is the file structure:

```
ICA-FastICA
    |-- src
        |-- FastICA.m
        |-- myICA.m
        |-- myICA2.m
        |-- myWhite.m
    |-- data
        |-- several audio files
    |-- icatest.m
    |-- Readme.md
    |-- 学习笔记 _ 独立成分分析(ICA, FastICA)及应用.md
```
Among the files above:
- In folder 'src';
  - file 'FastICA.m' is the file to realize the FastICA algorithm;
  - file 'myICA.m' is the file to realize the ICA algorithm where the number of spokers is 2;
  - file 'myICA2.m' is the file to realize the ICA algorithm where the number of spokers is more than 2;
  - file 'myWhite.m' is the file for data whitening;
- in folder 'voice'
  - there are several audio files in it;
- file 'icatest.m' is the test file to examine the performance of ICA and FastICA;
- file '学习笔记 _ 独立成分分析(ICA, FastICA)及应用.md' is a detailed introduction document for this project. 

For more detailed information, refer to article [学习笔记 _ 独立成分分析(ICA, FastICA)及应用.md]().
