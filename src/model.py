from sklearn.naive_bayes import MultinomialNB       # 朴素贝叶斯算法
import pickle                                       # 数据加载
import joblib
import os
import jieba
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score # 评价指标

# 将当前目录设置为工作目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 结巴不输出日志
jieba.setLogLevel(jieba.logging.INFO)

def model_train():
    print('>>> 模型训练中......')
    # 1.加载训练数据
    train_data = pickle.load(open('../data/processed/train.pkl', 'rb'))
    # 2.初始化模型
    model = MultinomialNB()
    model.fit(train_data['X'], train_data['y'])
    # 3.模型存储
    joblib.dump(model, '../experiments/checkpoints/邮件分类模型.pth')
    print('>>> 模型训练完成')

def model_evaluate():
    # 1.加载测试集 和 模型文件
    test_data = pickle.load(open('../data/processed/test.pkl', 'rb'))
    model = joblib.load('../experiments/checkpoints/邮件分类模型.pth')
    # 2.模型评估
    y_true = test_data['y']
    y_pred = model.predict(test_data['X'])
    print(model.score(test_data['X'], test_data['y']))
    print(f'准确率{accuracy_score(y_true, y_pred)}')
    print(f'精确率{precision_score(y_true, y_pred,pos_label="spam")}')
    print(f'召回率{recall_score(y_true, y_pred,pos_label="spam")}')

if __name__ == '__main__':
    # model_train()
    model_evaluate()
    pass