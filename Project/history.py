
# coding: utf-8

# In[1]:

import pandas as pd
MyLife=pd.read_csv('Myfile.tsv', sep='\t',names = ["URL","Host","Domain","Visit_time(ms)","Visit_time"
                                                  ,"Day_of_week","Transition_type","Title"])


# In[2]:

MyLife.head()


# In[3]:

MyLife.info()


# In[4]:

Mylife=MyLife.dropna(axis=0,how='any')


# In[5]:

Mylife.isnull().sum()


# In[6]:

Mylife['Date'],Mylife['Time']=Mylife.Visit_time.str.split(' ').str
def format_time(Mylife):
    Mylife.Time = Mylife.iloc[:,-1].apply(lambda x: x[0:8])
format_time(Mylife)


# In[7]:

Mylife.drop('Visit_time',axis=1)


# In[8]:

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
graph=sns.countplot(x='Day_of_week',data=Mylife)
plt.show()


# In[9]:

dm=pd.value_counts(Mylife.Host).head(20)
dm =dm.to_frame()
dm=dm.reset_index(drop=False)
dm.columns = ['Domain','Count']
dm.head(10)


# In[10]:

a=sns.kdeplot(dm.Count)
plt.show()


# In[ ]:




# In[11]:

ax=sns.barplot(x='Domain',y='Count',data=dm,palette='Blues')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show()


# In[12]:

Mylife.Title


# In[13]:

from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
text=Mylife['Title'].to_string()
# wc = WordCloud(max_font_size = 100)
# process_word = WordCloud.process_text(wc,text)
# sort = sorted(process_word.items(),key=lambda e:e[1],reverse=True)
# print(sort[:50])
text


# In[14]:

from PIL import *
import PIL.Image


# In[ ]:




# In[ ]:




# In[15]:

import chardet
import jieba
import numpy as np
# text=' '.join(jieba.cut(text,cut_all=False))


# In[16]:

background_Image = np.array(Image.open('3333.png'))
img_colors = ImageColorGenerator(background_Image)
# font_path = '../System/Library/Fonts/SFNSDisplay-SemiboldItalic.otf '


# In[17]:


stopwords=STOPWORDS

wc = WordCloud(
        collocations=False,
        font_path='STHeiti Medium.ttc',
        margin = 1, # 页面边缘
        mask = background_Image,
        scale = 2,
        max_words = 200, # 最多词个数
        min_font_size = 4, #
        stopwords = stopwords,
        random_state = 42,
        background_color = 'white', # 背景颜色
#         background_color = '#C3481A', # 背景颜色
        max_font_size = 100,
        )
wc.generate(text)


# In[18]:

process_word = WordCloud.process_text(wc,text)
sort = sorted(process_word.items(),key=lambda e:e[1],reverse=True)
print(sort[:300])


# In[19]:

wc.recolor(color_func=img_colors)
# 存储图像
wc.to_file('IronMan.png')
# 显示图像
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()


# In[20]:

stopwords.add('Search')
stopwords.add('华人首家在线视频分享网站')
stopwords.add('澳大利亚华人首家在线视频分享网站')
stopwords.add('收件箱')
stopwords.add('澳洲同城影视网')


# In[ ]:




# In[ ]:



