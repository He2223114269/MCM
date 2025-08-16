import jieba
import os
import  numpy as np
import jieba.posseg as pseg
wordslist = []
titlelist = []
"""
# 遍历文件夹
for file in os.listdir('.'):
    if '.' not in file:
        # 遍历文档
        for f in os.listdir(file):
            # 标题
            # windows下编码问题添加：.decode('gbk', 'ignore').encode('utf-8'))
            titlelist.append(file+'--'+f.split('.')[0])
            # 读取文档
            with open(file + '//' + f, 'r') as f:
                content = f.read().strip().replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')
            # 分词
            seg_list = jieba.cut(content, cut_all=True)
            result = ' '.join(seg_list)
            wordslist.append(result)

stop_word = [str(line.rstrip()) for line in open('stop_words.txt')]

...
seg_list = jieba.cut(content, cut_all=True)
seg_list_after = []
# 去停用词
for seg in seg_list:
    if seg.word not in stop_word:
        seg_list_after.append(seg)
result = ' '.join(seg_list_after)
wordslist.append(result)

vectorizer = CountVectorizer()
word_frequence = vectorizer.fit_transform(wordslist)
words = vectorizer.get_feature_names()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(word_frequence)
weight = tfidf.toarray()
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer

def titlelist():
    for file in os.listdir('.'):
        if '.' not in file:
            for f in os.listdir(file):
                yield (file+'--'+f.split('.')[0]).decode('gbk', 'ignore').encode('utf-8') # windows下编码问题添加：.decode('gbk', 'ignore').encode('utf-8'))

def wordslist():
    # jieba.add_word(u'丹妮莉丝')
    stop_word = [str(line.rstrip()) for line in open('stop_words.txt',encoding='utf-8')]
    print (len(stop_word))
    for file in os.listdir('.'):
        if '.' not in file:
            for f in os.listdir(file):
                with open(file + '//' + f) as t:
                    content = t.read().strip().replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')
                    seg_list = pseg.cut(content)
                    seg_list_after = []
                    #
                    for seg in seg_list:
                        if seg.word not in stop_word:
                            seg_list_after.append(seg.word)
                    result = ' '.join(seg_list_after)
                    # wordslist.append(result)
                    yield result


if __name__ == "__main__":

    wordslist = list(wordslist())
    titlelist = list(titlelist())

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(wordslist))

    words = vectorizer.get_feature_names()  #所有文本的关键字
    weight = tfidf.toarray()

    print('ssss')
    n = 5 # 前五位
    for (title, w) in zip(titlelist, weight):
        print(u'{}:'.format(title))
        # 排序s
        loc = np.argsort(-w)
        for i in range(n):
            print(u'-{}: {} {}'.format(str(i + 1), words[loc[i]], w[loc[i]]))
        print('\n')