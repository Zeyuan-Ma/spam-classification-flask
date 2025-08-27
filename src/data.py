# 数据处理相关
import pandas as pd             # 数据清洗
import os                       # 工作路径
import codecs                   # 编码格式
import re                       # 正则表达式
import zhconv                   # 繁体转简体
import jieba                    # 分词
import jieba.posseg as psg      # 词性筛选
# 模型训练相关
from sklearn.feature_extraction.text import CountVectorizer # 向量计数
from sklearn.model_selection import train_test_split        # 划分数据集
# 其他
import pickle           # 模型保存与导入
import time             # 时间
from tqdm import tqdm   # 进度条显示

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # 将当前目录设置为工作目录,即test1这个文件目录
jieba.setLogLevel(jieba.logging.INFO)                   # 结巴不输出日志

def load_email_data():
    """
    通过分析数据集，把文件进行读取、分割、数据集训练集切分
    组合成两个大的csv文件
    """
    # 读取index文件下的标签和文件路径
    labels,filenames = [],[]
    with open('../data/raw/full/index') as f:
        for line in f:
            label, path = line.strip().split()
            labels.append(label)
            filenames.append(path)

    # 根据上一步得到的文件路径，读取文件内容
    context =  []
    os.chdir('../data/raw/full/')
    for filename in filenames:
        with open(filename,encoding='gbk',errors='ignore') as f:
            context.append(f.read())

    # 切分数据集  训练集特征/测试集特征 / 训练集标签/测试集标签          特征/标签
    x_train,x_test,y_train,y_test = train_test_split(context,labels,test_size=0.2,random_state=42)

    # 存储到csv文件
    os.chdir('../../merge')
    pd.DataFrame({'label':y_train,'context':x_train}).to_csv('raw_train.csv',index=False)
    pd.DataFrame({'label':y_test,'context':x_test}).to_csv('raw_test.csv',index=False)

# 测试
if __name__ == '__main__':
    # print(os.getcwd())
    # load_email_data()
    os.chdir('../data/merge')
    df = pd.read_csv('raw_train.csv')
    df2 = pd.read_csv('raw_test.csv')
    print(df)
    print(df2)