import pandas as pd             # 数据清洗
import os                       # 工作路径
import codecs                   # 编码格式
import re                       # 正则表达式
from opencc import OpenCC       # 繁体转简体
import jieba                    # 分词
import jieba.posseg as pseg     # 词性筛选
from tqdm import tqdm           # 显示进度

from sklearn.feature_extraction.text import CountVectorizer # 文字特征提取：向量计数
from sklearn.model_selection import train_test_split        # 划分数据集

import pickle           # 数据保存与导入

os.chdir(os.path.dirname(os.path.abspath(__file__)))    # 将当前目录设置为工作目录,即test1这个文件目录
cc = OpenCC('t2s')                                      # 繁体转简体
jieba.setLogLevel(jieba.logging.INFO)                   # 结巴不输出日志
tqdm.pandas()                                           # tqdm 和 pandas 集成

def load_email_data():
    """
    通过分析数据集，把文件进行读取、分割、组合成大的csv文件
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
    os.chdir('../data/raw/full/') # 因为filename的路径是基于full下的，所以要改一下工作目录
    for filename in filenames:
        with open(filename,encoding='gbk',errors='ignore') as f:
            context.append(f.read())

    # 合并 存储到csv文件
    os.chdir('../../merge')
    pd.DataFrame({'context':context,'label':labels}).to_csv('data_raw.csv',index=False)


def clean_text(text):
    """
    对文本进行清洗，包括：1. 去掉非中文字符2. 繁体转简体3. 分词 + 词性筛选
    """
    if not isinstance(text, str):
        return ""
    # 1. 去掉非中文字符
    text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    # 2. 繁体转简体
    text = cc.convert(text)
    # 3. 分词 + 词性筛选
    allowed_pos = {"n", "v", "a", "nr", "ns", "nt"}
    words = pseg.cut(text)
    return " ".join([word for word, flag in words if flag in allowed_pos])


def feature_process():
    """
    对数据进行特征处理，包括：1. 统计特征 2. 词袋模型特征 3. 词向量特征
    """
    # 加载要处理的数据
    data_clean = pd.read_csv('../data/clean/data_clean.csv')
    # 确保 context 没有 NaN
    data_clean['context'] = data_clean['context'].fillna('')
    # 统计特征
    transfer = CountVectorizer(max_features=10000)
    x = transfer.fit_transform(data_clean['context'])
    y = data_clean['label']

    # 切分数据集   训练集特征/测试集特征 / 训练集标签/测试集标签                 特征/标签
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
    # 保存训练集和测试集 这里保存为pkl最合适，因为X是稀疏矩阵，csv不方便存
    train_data = {'X': x_train, 'y': y_train}
    test_data = {'X': x_test, 'y': y_test}
    pickle.dump(train_data, open('../data/processed/train.pkl', 'wb'))
    pickle.dump(test_data, open('../data/processed/test.pkl', 'wb'))
    # 保存词袋特征名，方便解释
    feature_names = transfer.get_feature_names_out()
    pickle.dump(feature_names, open('../data/processed/feature_names.pkl', 'wb'))


# 测试
if __name__ == '__main__':
    # 1.加载合并数据
    # load_email_data()
    # 查看一下数据
    # os.chdir(os.path.dirname(os.path.abspath(__file__))) # 回到当前.py所在目录
    # df = pd.read_csv('../data/merge/data_raw.csv')
    # print(df.head())

    # 2.对 context 列进行清洗，并显示进度,查看结果
    # 拷贝一份、清洗、保存
    # df_clean = df.copy()
    # df_clean["context"] = df_clean["context"].progress_apply(clean_text)
    # df_clean.to_csv("../data/clean/data_clean.csv", index=False, encoding="utf-8-sig")
    # 查看清洗后的结果
    # df2 = pd.read_csv('../data/clean/data_clean.csv')
    # print(df2.head())

    # 3.特征处理
    # feature_process()
    pass






