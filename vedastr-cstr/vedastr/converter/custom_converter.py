import torch

from .registry import CONVERTERS
from .base_convert import BaseConverter

# special token 정의
START = "<SOS>" # 문장 시작
END = "<EOS>"   # 문장 끝
PAD = "<PAD>"   # 패딩
SPECIAL_TOKENS = [START, END, PAD]

def load_vocab(tokens_paths):
    '''
    토큰 txt 파일을 불러와 토큰-id 딕셔너리 생성
    - `token_paths`: 토큰 txt 파일 경로
    '''
    tokens = [] # 토큰 저장 공간
    tokens.extend(SPECIAL_TOKENS)   # special 토큰 추가

    # 파일을 불러와 토큰 추출
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            for token in reader.split("\n"):
                if token not in tokens:
                    tokens.append(token)
    tokens.remove('')   # 빈 문자열 제거
    token_to_id = {tok: i for i, tok in enumerate(tokens)}  # 토큰 to id
    id_to_token = {i: tok for i, tok in enumerate(tokens)}  # id to 토큰
    return tokens, token_to_id, id_to_token

@CONVERTERS.register_module
class CustomConverter(BaseConverter):

    def __init__(self, character, batch_max_length=25):
        token_path = ["/opt/ml/input/data/train_dataset/tokens.txt"]    # 토큰 파일 경로
        self.character, self.dict, self.id_to_token = load_vocab(token_path)    # 토큰 파싱
        self.batch_max_length = batch_max_length    # 최대 시퀀스 길이 설정
        self.ignore_index = self.dict[PAD]  # 무시할 토큰 지정
    

    def train_encode(self, text_list):
        '''
        배치 크기의 시퀀스들을 인코딩
        - `text_list`: 배치 크기의 시퀀스 리스트
        '''
        length = [len(s) + 1 for s in text_list]  # batch 안에 있는 ground truth의 길이 저장
        batch_text = torch.LongTensor(len(text_list), self.batch_max_length+1).fill_(self.ignore_index)  # 인코딩 행렬 생성 (b, max_len+1)
        for i, t in enumerate(text_list):
            text = t.split(' ') 
            text.append(END)    # EOS 추가
            text = [self.dict[char] for char in text]   # 토큰 to id로 변환
            batch_text[i][:len(text)] = torch.LongTensor(text)  # 행렬에 저장
        batch_text_input = batch_text
        batch_text_target = batch_text

        return batch_text_input, torch.IntTensor(length), batch_text_target


    def decode(self, text_index):
        '''
        배치 크기의 id 시퀀스를 토큰 시퀀스로 디코딩
        - `text_index`: 배치 크기의 id 시퀀스 리스트
        '''
        texts = []
        batch_size = text_index.shape[0]
        for index in range(batch_size):
            text = ' '.join([self.id_to_token[int(i)] for i in text_index[index, :]])
            text = text[:text.find(END)]    # eos까지만 가져오기
            texts.append(text) 

        return texts
