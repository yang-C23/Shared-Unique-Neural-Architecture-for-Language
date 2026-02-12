import json
import shutil

"""
由于Mistral-NeMo-Minitron-8B-Base的词表大小为131072，因此存储的时候使用np.uint32存储，一个数字占4个字节，范围在0~2^32-1
    每个sample的长度不固定
    索引需要记录每个句子的起始位置和长度
    句子的起始位置用np.uint64存储，8B，
    长度用np.uint32存储，4B 最大为4294967295
"""
"""
合并不同的数据集：
    1. 如果是同类型的，则直接把两个文件拼接起来即可
    2. 如果是不同类型的，则需要额外生成一个文件，称为.dist，这个文件存储了每个类型的样本总数
    暂定.dist文件为torch.save保存的List文件
"""
"""
warmup: https://stackoverflow.com/questions/11832254/understanding-performance-of-numpy-memmap
"""

import numpy as np
import os
from transformers import AutoTokenizer
from typing import List
import argparse
import multiprocessing
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset
import re

"""try-catch来自于Megatron-Deepspeed/tools/preprocess_data.py"""
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):
    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class ChineseSplitter(object):
    def __init__(self, pattern):
        self.pattern = pattern
        if "(" in self.pattern and ")" in self.pattern:
            """保留分隔符"""
            self.keep = True
        else:
            self.keep = False

    def tokenize(self, text):
        if self.keep:
            new_list = []
            _list = re.split(pattern=self.pattern, string=text)
            for item in _list:
                if len(new_list) == 0:
                    """如果是第一个元素，则直接进行一个保留"""
                    new_list.append(item)
                elif (len(item) == 1 and item in self.pattern) or (item == "\n" or item == r"\n") or len(item) == 0:
                    """说明是分隔符"""
                    new_list[-1] += item
                else:
                    new_list.append(item)
        else:
            new_list = re.split(pattern=self.pattern, string=text)
        return new_list

"""refer: https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/data/indexed_dataset.py#L349"""
def _warmup_mmap_file(path):
    return
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass

"""Mistral-NeMo-Minitron-8B-Base的分词器"""
class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        self.sp_model = AutoTokenizer.from_pretrained(model_path)
        
        # BOS / EOS token IDs for Mistral-NeMo-Minitron-8B-Base
        self.n_words: int = self.sp_model.vocab_size
        self.eos_id: int = self.sp_model.eos_token_id  # 2
        self.bos_id: int = self.sp_model.bos_token_id  # 1
        self.unk_id: int = self.sp_model.unk_token_id if hasattr(self.sp_model, 'unk_token_id') else 0
        # Mistral通常使用EOS token作为PAD token
        self.pad_id: int = self.sp_model.pad_token_id if self.sp_model.pad_token_id is not None else self.eos_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str, print(f"type:{type(s)},content:{s}")
        t = self.sp_model.encode(s, add_special_tokens=False)  # 手动控制特殊token
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

class DistributedTokenizer:
    def __init__(self, args, eos: bool, bos: bool, collate_fn=None):
        self.args = args
        self.max_seq_length = self.args.seq_length
        self.eos = eos
        self.bos = bos
        self.collate_fn = collate_fn

    def split(self, lst: List[int]):
        '''
        修改后的版本:
        如果这个句子超过最大长度seq_length 将其丢弃
        '''
        maxlen = self.max_seq_length
        merged_lst = []
        i = j = 0
        answer_lst = []
        while i < len(lst):
            ans = [0, 0]
            ans[0] = i
            sums = lst[i]
            j = i + 1
            while j < len(lst) and sums + lst[j] <= maxlen:
                sums += lst[j]
                j += 1
            k = j  # 记录一下终点(不包括这个点)
            ans[1] = j
            i = k  #
            merged_lst.append(sums)
            answer_lst.append(ans)
        if len(merged_lst) >= 2 or sums > maxlen:
            print(f"one exceed max seq_len, sums:{sums}, merged_lst length:{len(merged_lst)}")
            merged_lst=[]
            answer_lst=[]
            
        return merged_lst,answer_lst  # 左闭右开

    def dsmt_initializer(self):
        """加载分词器"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)
        if self.args.language.lower() == "english":
            if self.args.do_split_sentences:
                if not nltk_available:
                    print("NLTK is not available to split sentences.")
                    exit()
                splitter = nltk.load("tokenizers/punkt/english.pickle")
                if self.args.do_keep_newlines:
                    DistributedTokenizer.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                        train_text=splitter._params,
                        lang_vars=CustomLanguageVars())
                else:
                    DistributedTokenizer.splitter = splitter
            else:
                DistributedTokenizer.splitter = IdentitySplitter()
        elif self.args.language.lower() == "french":
            if self.args.do_split_sentences:
                pattern = r"([.?!。\n])"
                DistributedTokenizer.splitter = ChineseSplitter(pattern=pattern)
            else:
                DistributedTokenizer.splitter = IdentitySplitter()
        elif self.args.language.lower() == "chinese":
            if self.args.do_split_sentences:
                if self.args.do_keep_newlines:
                    pattern = r"([;!?；？。！\n])"
                    DistributedTokenizer.splitter = ChineseSplitter(pattern=pattern)
                else:
                    pattern = r"[;!?；？。！\n]"
                    DistributedTokenizer.splitter = ChineseSplitter(pattern=pattern)
            else:
                DistributedTokenizer.splitter = IdentitySplitter()
        else:
            assert False, "目前支持的语言为english、chinese和french，请确保输入正确"

    def _re_split(self, src: str, tokenized: List, start_part=False, end_part=False):
        """
        :param src:         原始的句子
        :param tokenized:   分完词后的列表
        :param start_part:  传入的src为开始部分，说明当前tokenized的开头有BOS
        :param end_part:    传入的src为结束部分，说明当前tokenized的结尾有EOS
        :return:
        """
        
        if len(tokenized) <= self.max_seq_length:
            return [tokenized]
        else:
            """超出最大长度"""
            n_block = int(np.ceil(len(tokenized) / self.max_seq_length).item())
            if self.args.language.lower() == "english":
                """英文就直接对tokenized均分"""
                new_tokenized = []
                for i in range(n_block):
                    new_tokenized.append(tokenized[i * self.max_seq_length:(i + 1) * self.max_seq_length])
            elif self.args.language.lower() == "chinese":
                """中文需要对src进行分割，然后送入到tokenize进入"""
                new_tokenized = []
                if len(tokenized) % self.max_seq_length >= self.max_seq_length * 0.8:
                    n_block += 1
                length_per_block = int(np.ceil(len(src) / n_block).item())
                for i in range(n_block):
                    new_src = src[i * length_per_block:(i + 1) * length_per_block]
                    bos = True if i == 0 and start_part == True and self.bos else False
                    eos = True if i == n_block - 1 and end_part == True and self.eos else False
                    _tokenized = DistributedTokenizer.tokenizer.encode(new_src, bos=bos, eos=eos)
                    new_tokenized.append(_tokenized)
            elif self.args.language.lower() == "french":
                """法语处理，类似英文"""
                new_tokenized = []
                for i in range(n_block):
                    new_tokenized.append(tokenized[i * self.max_seq_length:(i + 1) * self.max_seq_length])
            else:
                assert False, "目前支持的语言为english、chinese和french，请确保输入正确"
            return new_tokenized

    def dsmt_encode(self, json_line):
        if self.collate_fn == None:
            text = json_line
        else:
            text = self.collate_fn(json_line)
        
        if text == "\n" or text.strip() == "" or text == r"\n":
            return []
        
        """将其切分成句子"""
        sentences = DistributedTokenizer.splitter.tokenize(text)
        # 默认是Indientity splitter， 因此长度会为1，只有一句话
        assert len(sentences) == 1

        """对句子进行分词，然后对于超过长度的再次分割，处理完成之后送入split进行融合即可"""
        if len(sentences) == 1:
            # 只有一句话
            _tokenized = DistributedTokenizer.tokenizer.encode(sentences[0], bos=self.bos, eos=self.eos)
            _tokenized = [_tokenized]
        else:
            _tokenized = []
            for idx, sentence in enumerate(sentences):
                cur_tokenized = DistributedTokenizer.tokenizer.encode(sentence, bos=(idx == 0 and self.bos),
                                                                      eos=(idx == len(sentences) - 1) and self.eos)
                _tokenized.extend(
                    self._re_split(src=sentence, tokenized=cur_tokenized, start_part=(idx == 0) and self.bos,
                                   end_part=(idx == len(sentences) - 1) and self.eos))
        
        """记录下分句后每个句子token数目"""
        length_tokenized = [len(_) for _ in _tokenized]

        """得到合并的索引"""
        _,index = self.split(length_tokenized)
        ultra = []
        for pair in index:
            cur = []
            start, end = pair
            for i in range(start, end):
                cur.extend(_tokenized[i])
            ultra.append(cur)
        return ultra

    def initializer(self):
        """加载分词器"""
        DistributedTokenizer.tokenizer = Tokenizer(self.args.tokenizer_path)

    def encode(self, text: str):
        return DistributedTokenizer.tokenizer.encode(text.strip(), self.bos, self.eos)

class MyDataset(Dataset):
    def __init__(self, data_prefix, seq_length, pad_id):
        super(MyDataset, self).__init__()
        self.idx_file_path = f"{data_prefix}.idx"
        self.bin_file_path = f"{data_prefix}.bin"
        self.dis_file_path = f"{data_prefix}.dis"
        self.seq_length = seq_length
        self.pad_id = pad_id

        self.index_start_pos = None
        self.index_length = None
        self._load_index()
        self._load_bin()
        self._load_dis()

    def _load_index(self):
        """文件所占的字节大小"""
        file_size = os.stat(self.idx_file_path).st_size
        """样本总数 - Updated for uint32 length storage"""
        assert file_size % 12 == 0  # 4B的length，8B的start pos
        self.total_sample = file_size // 12
        with open(self.idx_file_path, "rb") as f:
            self.index_start_pos = np.frombuffer(f.read(self.total_sample * 8), dtype=np.uint64).tolist()
            self.index_length = np.frombuffer(f.read(self.total_sample * 4), dtype=np.uint32).tolist()

    def _load_bin(self):
        """参考了Megatron-Deepspeed"""
        _warmup_mmap_file(self.bin_file_path)
        """以内存映射的方式进行加载大文件 - Updated for uint32 token storage"""
        self.bin_buffer = np.memmap(self.bin_file_path, dtype=np.uint32, mode='r')

    def _load_dis(self):
        """仅当有多种类别的数据混合有效"""
        self.distributed = torch.load(self.dis_file_path)
        if len(self.distributed) != 0:
            assert sum(self.distributed) == self.total_sample

    def __len__(self):
        return self.total_sample

    def __getitem__(self, idx):
        start_idx = self.index_start_pos[idx]
        length = self.index_length[idx]
        if idx + 1 < self.total_sample:
            assert start_idx + length == self.index_start_pos[idx + 1], \
                f"{start_idx + length}!={self.index_start_pos[idx + 1]}, idx={idx}"
        if length > self.seq_length:
            length = self.seq_length
        return self.bin_buffer[start_idx:start_idx + length].tolist()

def count_lines(path):
    """计算输入文件的行数"""
    print(path)
    with open(path, 'rb') as f:
        count = 0
        last_data = '\n'
        while True:
            data = f.read(1024 * 1024 * 1024)
            if not data:
                break
            count += data.count(b'\n')
            last_data = data
        if last_data[-1:] != b'\n':
            count += 1
    return count

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="write", type=str, help="有merge,write,read三种模式")
    parser.add_argument("--seq_length", default=512, type=int, help="最大长度")
    parser.add_argument("--language", default="chinese", type=str, help="english, chinese, french")
    parser.add_argument("--do_split_sentences", action="store_true", default=False, help="是否将文档划分成句子")
    parser.add_argument("--do_keep_newlines", action="store_true", default=False, help="划分的时候是否保留换行符")
    parser.add_argument("--file_path", default="examples.txt", type=str, help="源文件，每一行都是一个样本")
    parser.add_argument("--num_workers", default=1, type=int, help="并行处理数量")
    parser.add_argument("--tokenizer_path", default='nvidia/Mistral-NeMo-Minitron-8B-Base',type=str, help="Tokenizer文件")
    parser.add_argument("--save_prefix", default="train", type=str, help="保存的时候叫什么,索引文件会添加上.idx,数据文件添加上.bin")
    parser.add_argument("--save_path", default="path_to_save/", type=str, help="保存的位置，需要结尾为/")
    parser.add_argument("--num_per_doc", default=-1, type=int, help="每个文档保留多少个sample，如果为-1表示全部都要")
    parser.add_argument("--read_path_prefix", default="./hello", type=str,
                        help="读取的文件前缀，读取的时候会自动补全.idx/.bin/.dis")
    parser.add_argument("--merge_path_prefix", default=None, type=str, help="需要合并的文件前缀，['1','2','3']")
    parser.add_argument("--merge_path_type", default=None, type=str, help="如果不提供，则默认为同一类型的数据集，提供了，则以[1,1,0]这种格式给出")
    parser.add_argument("--new_path_prefix", default=None, type=str, help="如果为None，则自动从上面的文件中选取最大的进行合并，如果不为None，则自动")
    parser.add_argument("--save_mode", default=1, type=int, help="0-不存储索引文件 1-存储索引文件")

    return parser.parse_args()

def collate_fn_from_json(json_line: str):
    data = json.loads(json_line)
    total_text = data['content']
    return total_text

def collate_fn_from_text(text: str):
    return text

def write(args):
    """统计文件行数"""
    print(f"[{time.ctime()}] 开始统计行数")
    count = count_lines(args.file_path)
    print(f"[{time.ctime()}] 行数为:{count}")
    """打开文本文件"""
    fin = open(args.file_path, 'r', encoding='utf-8')
    """创建多进程"""
    encoder = DistributedTokenizer(args, eos=True, bos=False, collate_fn=collate_fn_from_text)
    pool = multiprocessing.Pool(args.num_workers, initializer=encoder.dsmt_initializer)
    """从输入流中进行读取"""
    encoded_samples = list(
        (tqdm(pool.imap(encoder.dsmt_encode, fin, 25), total=count, desc="读取进度"))
    )
    print(f"[{time.ctime()}] 读取完毕")
    """开始写入"""
    """起始位置:np.uint64: 8B"""
    """长度:np.uint32: 4B (updated for Mistral)"""
    """token:np.uint32: 4B (updated for Mistral)"""
    f_bin_out = open(f"{args.save_path}{args.save_prefix}.bin", "wb")
    encoded_samples = list(encoded_samples)
    pbar = tqdm(total=len(encoded_samples))
    start_pos = 0
    start = []
    length = []
    num_samples = 0
    flag = True
    g = torch.Generator()
    g.manual_seed(2023)
    statistic = [0, 0, 0, 0]  # 开头为bos,结尾为eos,没有,完整的一个句子
    
    for doc in encoded_samples:
        if args.num_per_doc == -1:
            idx = list(range(len(doc)))
            if flag:
                print("文档全部采样")
                flag = False
        else:
            if args.num_per_doc <= 2:
                idx = torch.randint(0, len(doc), [args.num_per_doc], generator=g).tolist()
            else:
                idx = [0, -1]
                if len(doc) > 2:
                    idx.extend((torch.randperm(len(doc) - 2, generator=g) + 1).tolist())
                idx = idx[args.num_per_doc]
            if flag:
                print("文档局部采样")
                flag = False
        for i in idx:
            target = doc[i]
            if len(target) == 0:
                continue
            num_samples += 1
            
            # 统计BOS/EOS token
            if len(target) > 0:
                if target[0] == 1 and target[-1] == 2:  # BOS=1, EOS=2 for Mistral
                    statistic[3] += 1
                elif target[0] == 1 and target[-1] != 2:
                    statistic[0] += 1
                elif target[0] != 1 and target[-1] == 2:
                    statistic[1] += 1
                else:
                    statistic[2] += 1
            
            # Write tokens as uint32
            f_bin_out.write(np.array(target, dtype=np.uint32).tobytes(order='C'))
            length.append(len(target))
            start.append(start_pos)
            start_pos += len(target)
        pbar.update(1)
    
    f_bin_out.close()
    f_idx_out = open(f"{args.save_path}{args.save_prefix}.idx", "wb")
    f_idx_out.write(np.array(start, dtype=np.uint64).tobytes(order='C'))
    f_idx_out.write(np.array(length, dtype=np.uint32).tobytes(order='C'))  # Updated to uint32
    f_idx_out.close()
    print(num_samples)
    dis = [num_samples]
    torch.save(dis, f"{args.save_path}{args.save_prefix}.dis")
    torch.save(statistic, f"{args.save_path}{args.save_prefix}.tmp")

def read(args):
    ds = MyDataset(args.read_path_prefix, seq_length=args.seq_length, pad_id=0)
    tokenizer = Tokenizer(model_path=args.tokenizer_path)
    bos_token = tokenizer.bos_id
    eos_token = tokenizer.eos_id
    print("BOS token:", bos_token)
    print("EOS token:", eos_token)

    print(f"长度为{len(ds)}")
    for i in range(len(ds)):
        if i == 20:
            break
        print(f"分句：{i}", tokenizer.decode(ds[i]))
    print(f"分布为:{ds.distributed}")

def merge(args):
    """合并数据集"""
    if args.merge_path_prefix == None:
        assert False
    else:
        merge_path_prefix = eval(args.merge_path_prefix)

    if args.merge_path_type == None:
        print(f"[{time.ctime()}] 合并的数据集属于同一类型")
        merge_path_type = None
    else:
        merge_path_type = eval(args.merge_path_type)

    if args.new_path_prefix == None:
        assert False
    new_path_prefix = args.new_path_prefix

    if merge_path_type != None:
        # Handle different types of datasets
        classifier_prefix = {}
        for idx, types in enumerate(merge_path_type):
            if types not in classifier_prefix:
                classifier_prefix[types] = [merge_path_prefix[idx]]
            else:
                classifier_prefix[types].append(merge_path_prefix[idx])
        
        new_file_bin = open(new_path_prefix + ".bin", "wb")
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                with open(file_prefix + ".bin", "rb") as f:
                    shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        
        new_file_idx = open(new_path_prefix + ".idx", "wb")
        index_start_pos = []
        index_length = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                file_size = os.stat(file_prefix + ".idx").st_size
                assert file_size % 12 == 0  # Updated for uint32 length
                total_sample = file_size // 12
                with open(file_prefix + ".idx", "rb") as f:
                    _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                    _index_length = np.frombuffer(f.read(total_sample * 4), dtype=np.uint32).tolist()  # Updated to uint32
                if len(index_start_pos) > 0:
                    index_start_pos.extend((_index_start_pos + index_start_pos[-1] + index_length[-1]).tolist())
                else:
                    index_start_pos.extend(_index_start_pos)
                index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint32).tobytes(order='C'))  # Updated to uint32
        new_file_idx.close()
        
        _cur_size = 0
        new_dist = []
        for types, file_prefixes in classifier_prefix.items():
            for file_prefix in file_prefixes:
                data = torch.load(file_prefix + ".dis")
                assert len(data) == 1
                _cur_size += data[0]
            new_dist.append(_cur_size)
            _cur_size = 0
        torch.save(new_dist, new_path_prefix + ".dis")
        assert sum(new_dist) == len(index_start_pos)
    else:
        # Handle same type datasets
        new_file_bin = open(new_path_prefix + ".bin", "wb")
        for file in merge_path_prefix:
            with open(file + ".bin", "rb") as f:
                shutil.copyfileobj(f, new_file_bin)
        new_file_bin.close()
        
        new_file_idx = open(new_path_prefix + ".idx", "wb")
        index_start_pos = []
        index_length = []
        for file in merge_path_prefix:
            file_size = os.stat(file + ".idx").st_size
            assert file_size % 12 == 0  # Updated for uint32 length
            total_sample = file_size // 12
            with open(file + ".idx", "rb") as f:
                _index_start_pos = np.frombuffer(f.read(total_sample * 8), dtype=np.uint64)
                _index_length = np.frombuffer(f.read(total_sample * 4), dtype=np.uint32).tolist()  # Updated to uint32
            if len(index_start_pos) > 0:
                index_start_pos.extend((_index_start_pos + index_start_pos[-1] + index_length[-1]).tolist())
            else:
                index_start_pos.extend(_index_start_pos)
            index_length.extend(_index_length)
        new_file_idx.write(np.array(index_start_pos, dtype=np.uint64).tobytes(order='C'))
        new_file_idx.write(np.array(index_length, dtype=np.uint32).tobytes(order='C'))  # Updated to uint32
        new_file_idx.close()
        assert len(index_start_pos) == len(index_length)
        
        torch.save([len(index_start_pos)], new_path_prefix + ".dis")
        total = 0
        for file in merge_path_prefix:
            data = torch.load(file + ".dis")
            assert len(data) == 1
            total += data[0]
        assert total == len(index_start_pos)

if __name__ == '__main__':
    args = get_args()
    print(args)
    if args.mode.lower() == "read":
        read(args)
    elif args.mode.lower() == "write":
        write(args)
    elif args.mode.lower() == "merge":
        merge(args)
    else:
        assert False