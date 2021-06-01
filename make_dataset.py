%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import StratifiedKFold
InteractiveShell.ast_node_interactivity = "all"

'''
    현재 txt로 저장할때 " 토큰이 """로 변환되서 저장되는 버그(?)가 있습니다. 
'''

def create_data_frame(default_image_path:str,data_path:str,level_path:str,source_path:str):
    '''
        데이터 프레임 만들기
        이미지 평균을 내야되서 시간이 좀 많이 걸립니다.
    '''
    data = {}
    data['latex'] = []
    data['seq_len'] = []
    data['aspect_ratio'] = []
    data['image_width'] = []
    data['image_height'] = []
    data['level']=[]
    data['source']=[]
    data['grey_mean']=[] # 이미지 gray scale로 바꾼것의 평균 (밝기의 지표)
    data['dummy']=[] # StratifiedKFold를 위한 dummy target 값
    all_latex_list = []
    lv1_latex_list=[]
    lv2_latex_list=[]
    lv3_latex_list=[]
    lv4_latex_list=[]
    lv5_latex_list=[]
    
    with open(level_path) as f:
        level_info={}
        for line in f:
            path,level=line.replace("\n","").split("\t")
            level_info[path]=int(level)
            
    with open(source_path) as f:
        source_info={}
        for line in f:
            path,source=line.replace("\n","").split("\t")
            source_info[path]=str(source)
            
    with open(data_path) as f:
        for idx,line in enumerate(f):
            image_path,latex=line.replace("\n","").split("\t")
            image = Image.open(default_image_path+image_path).convert('L')
            width, height = image.size
            data['aspect_ratio'].append(round(width / height,1))
            data['image_width'].append(int(width))
            data['image_height'].append(int(height))
            latex=latex.split(" ")
            data['latex'].append(latex)
            data['seq_len'].append(len(latex))
            level=level_info[image_path]
            data['level'].append(level)
            source=source_info[image_path]
            data['source'].append(source)
            data['grey_mean'].append(np.mean(image))
            data['dummy'].append(0)
            all_latex_list += latex
            if level==1:
                lv1_latex_list += latex
            elif level==2:
                lv2_latex_list += latex
            elif level==3:
                lv3_latex_list += latex
            elif level==4:
                lv4_latex_list += latex
            else:
                lv5_latex_list += latex
    df = pd.DataFrame.from_dict(data)
    lv_latex_list=[lv1_latex_list,lv2_latex_list,lv3_latex_list,lv4_latex_list,lv5_latex_list]
    return df, all_latex_list,lv_latex_list
def split_dataset(df,n_splits):
    '''
        input : dataframe, 몇개로 나눌건지
        outpur : (train_index, test_index) 
    '''
    skf=StratifiedKFold(n_splits=n_splits)
    return next(iter(skf.split(df,df['dummy'])))

def convert_filename(x):
    return f'train_{str(x).zfill(5)}.jpg'

def list_latex2str(x):
    return ' '.join(x)

def plot_dist(df, field, bins, color, xlabel, ylabel, title):
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18,6))
    sns.distplot(df[field], bins=bins, color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    plt.show()
df, all_latex_list, lv_latex_list = create_data_frame(default_image_path="/opt/ml/input/data/train_dataset/images/",\
       data_path="/opt/ml/input/data/train_dataset/gt.txt",level_path="/opt/ml/input/data/train_dataset/level.txt",source_path="/opt/ml/input/data/train_dataset/source.txt")
# 원본 dataframe(100,000)개를 8 : 2 로 쪼개기
train_idx,test_idx=split_dataset(df,5)
train_dataset=df.iloc[train_idx]
test_dataset=df.iloc[test_idx]

# 앞서 나눠진 train_dataset 80,000개를 64,000 , 16,000으로 쪼개기
dummy_idx,real_train_idx=split_dataset(train_dataset,5)
real_train_dataset=train_dataset.iloc[real_train_idx]
# 앞서 나눠진 test_dataset 20,000개를 16,000 , 4,000으로 쪼개기
dummy_idx,real_test_idx=split_dataset(test_dataset,5)
real_test_dataset=test_dataset.iloc[real_test_idx]

# 파일이름, latex를 ground truth 형식으로 바꾸기
real_train_dataset['filename']=real_train_dataset.index.map(convert_filename)
real_train_dataset['latex_str']=real_train_dataset['latex'].map(list_latex2str)

real_test_dataset['filename']=real_test_dataset.index.map(convert_filename)
real_test_dataset['latex_str']=real_test_dataset['latex'].map(list_latex2str)

# 파일 저장
real_train_dataset[['filename','latex_str']].to_csv('custom_train.txt', sep = '\t', header=False, index = False)
real_test_dataset[['filename','latex_str']].to_csv('custom_test.txt', sep = '\t', header=False, index = False)

# 데이터 분포 visualize
for name in real_train_dataset.columns.tolist()[2:]:
    plot_dist(df=real_train_dataset, field=name.strip(), bins=50, color='b', xlabel='Sequence Length', \
        ylabel='Frequency', title=name+'_train')
    plot_dist(df=real_test_dataset, field=name, bins=50, color='b', xlabel='Sequence Length', \
        ylabel='Frequency', title=name+'_test')
# #%%
# real_train_dataset[real_train_dataset['filename']=='train_35487.jpg']
# # %%
# # import csv
# # real_test_dataset[['filename','latex_str']].to_csv('custom_test.txt', sep = '\t', header=False, index = False, quoting=csv.QUOTE_NONE, encoding='utf-8')
# # %%
