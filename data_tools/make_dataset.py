# %matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from PIL import Image
from IPython.core.interactiveshell import InteractiveShell
from sklearn.model_selection import StratifiedKFold
InteractiveShell.ast_node_interactivity = "all"

'''
    현재 txt로 저장할때 " 토큰이 """로 변환되서 저장되는 버그가 있습니다. 
'''

def create_data_frame(default_image_path:str,data_path:str,level_path:str,source_path:str):
    """Make DatFrame
       It takes few minutes to calculate imgae brigtness average 

    Args:
        default_image_path (str):  Image path (It must end with '/')
        data_path (str): gt.txt path
        level_path (str): level.txt path
        source_path (str): source.txt path

    Returns:
        df (Dataframe): Dataframe of dataset.
    """
    data = {}
    data['latex'] = [] # groung truth
    data['seq_len'] = [] # ground truth 길이
    data['aspect_ratio'] = [] # 이미의 가로 세로의 비율
    data['image_width'] = [] # 가로 길이
    data['image_height'] = [] # 세로 길이
    data['level']=[] # 이미지 레벨
    data['source']=[] # 이미지가 손글씨인지, 컴퓨토 이미지인지
    data['grey_mean']=[] # 이미지 gray scale로 바꾼것의 평균 (밝기의 지표)
    data['dummy']=[] # StratifiedKFold를 위한 dummy target 값
    
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
        for line in f:
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
    df = pd.DataFrame.from_dict(data)
    return df


def split_dataset(df,n_splits):
    """Stratified K-Folds cross-validator.
        Provides first train/test indices to split data in train/test sets.

    Args:
        df (Dataframe) : datafram to split
        n_splits (int) : Number of folds. Must be at least 2.

    Returns:
        (train_index, test_index) (tuple) :
                train_index : The first training set indices for that split.
                test_index : The first testing set indices for that split.
    """
    skf=StratifiedKFold(n_splits=n_splits)
    return next(iter(skf.split(df,df['dummy'])))


def convert_filename(x):
    """Dataframe's map function 
        Replaces the image number of the custom dataset to ground truth format.

    Args:
        x (int) : Image num to convert

    Returns:
        Filename of ground truth format(str)
    """
    return f'train_{str(x).zfill(5)}.jpg'


def list_latex2str(x):
    """Dataframe's map function 
        Replaces the latex of the custom dataset to ground truth format.

    Args:
        x (list) : latex string to convert

    Returns:
        latex of ground truth format(str)
    """
    return ' '.join(x)


def plot_dist(df, field, bins, color, xlabel, ylabel, title):
    """Visualize DatFrame's column

    Args:
        df (Dataframe) :  Dataframe to visualize
        field (string) : The column of the Dataframe that want to see the distribution from.
        bins (int) : Number of bar graphs to show.
        color (string) : Color want to see.
        xlabel ((string) : x-axis label.
        ylabel (string) : y-axis label.
        title (string) : Graph title.
    Returns: 
            None
    """
    sns.set(color_codes=True)
    fig, ax = plt.subplots(figsize=(18,6))
    sns.distplot(df[field], bins=bins, color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=20)
    plt.show()


def create_small_dataset():
    """Make smaller dataset
        Save new train & test dataset in text file
        Args:
            root_dir(str) : root_dir path
            n_splits(int) : Number of folds. Must be at least 2.
            filename(str,optinal) : Filename to be saved
            is_visual(bool,optinal) : if True visualize dataframe plot
        Returns: 
            None
    """
    root_dir="/opt/ml/input/data/train_dataset"

    # 데이터 프레임 만들기
    df= create_data_frame(
        default_image_path=os.path.join(root_dir,'images/'),
        data_path=os.path.join(root_dir,'gt.txt'),
        level_path=os.path.join(root_dir,'level.txt'),
        source_path=os.path.join(root_dir,'source.txt'))

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

    ## 주피터 노트북이 필요
    # 데이터 분포 visualize 
    for name in real_train_dataset.columns.tolist()[2:]:
        plot_dist(df=real_train_dataset, field=name.strip(), bins=50, color='b', xlabel='Sequence Length', \
            ylabel='Frequency', title=name+'_train')
        plot_dist(df=real_test_dataset, field=name, bins=50, color='b', xlabel='Sequence Length', \
            ylabel='Frequency', title=name+'_test')


if __name__ == '__main__':  
    create_small_dataset()
   