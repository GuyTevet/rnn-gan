import os
import bz2
import logging
import shutil
from datetime import datetime
import json
import sys
from tqdm import tqdm
import time
import operator
import threading
import h5py
import numpy as np
import argparse
from copy import copy
import random
import collections

#local
import data


"""
Data_handler (base class)
"""
class Data_handler(object):

    def __init__(self,output_dir):

        self.output_dir = output_dir
        self.log_dir = os.path.join(self.output_dir,'log')

        #dirs & files
        date = datetime.now().strftime('%d-%m-%Y_%H:%M')
        self.log_file = os.path.join(self.log_dir,"log_%s.txt"%date)

    def text_block(self,files, size=65536):
        while True:
            b = files.read(size)
            if not b: break
            yield b

    def set_logger(self):

        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        format = '%(asctime)s\t[%(levelname)s]:\t%(message)s'
        logFormatter = logging.Formatter(format)
        logging.basicConfig(filename=self.log_file,
                            level=logging.DEBUG,
                            format=format)
        stream_hdl = logging.StreamHandler()
        stream_hdl.setFormatter(logFormatter)
        stream_hdl.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(stream_hdl)

    def logger_announce(self,str):
        logging.info('~*~*~*~*~*~~*~*~*~*~~*~*~*~')
        logging.info("DATA HANDLER - [%0s]"%str)
        logging.info('~*~*~*~*~*~~*~*~*~*~~*~*~*~')

    def merge_txt(self,source_list,target):

        with open(target, 'w') as outfile:
            for fname in source_list:
                with open(fname,'r') as infile:
                    for line in infile:
                        outfile.write(line)

    def merge_shuffle_txt(self,source_list,target):

        logging.info("[merge_shuffle_txt] %s to [%s]"%(str(source_list),target))

        #open all source files
        f_list = [open(f_path,'r') for f_path in source_list]

        with open(target, 'w') as outfile:
            while len(f_list) > 0 :
                random_file = random.choice(f_list)
                random_line = random_file.readline()
                if random_line != '':
                    outfile.write(random_line)
                else:
                    random_file.close()
                    f_list.remove(random_file)

    def prepare_data(self):

        #create output folder
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        #set logger
        self.set_logger()

"""
Reddit_data_handler
"""
class Reddit_data_handler(Data_handler):
    """
    preprocessing reddit comments data
    # downloaded from:
    https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/
    # torrent magnet:
    magnet:?xt=urn:btih:7690f71ea949b868080401c749e878f98de34d3d&dn=reddit%5Fdata&tr=http%3A%2F%2Ftracker.pushshift.
    io%3A6969%2Fannounce&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80
    """
    def __init__(self,input_dir,output_dir,subreddit_list,debug_mode=False):

        super(Reddit_data_handler, self).__init__(output_dir)
        self.input_dir = input_dir
        self.subreddit_list = subreddit_list
        self.debug_mode = debug_mode

    def extract_bz2_file(self,file_path,save_path):
        try:
            with open(save_path, 'wb') as new_file, bz2.BZ2File(file_path, 'rb') as file:
                for data in iter(lambda: file.read(1024 * 1024), b''):
                    new_file.write(data)
        except:
            logging.info("UNEXPECTED ERROR DURING EXTRACT OF [%0s] - SKIPPING"%file_path)

    def json_process(self,input_path,output_dir):

        hist_path = os.path.join(output_dir,'hist.json')

        #try open statistics histogram
        if self.debug_mode:
            if os.path.exists(hist_path):
                with open(hist_path,'r') as f:
                    hist = json.load(f)
            else:
                hist = {}

        with open(input_path,'r') as in_json:
            line = 'START'
            count = 0

            #process single comment
            while line != '':
                line = in_json.readline()
                try:
                    comment = json.loads(line)
                    subreddit = comment['subreddit']
                    body = comment['body']

                    #add comment if not empty
                    if str(subreddit) in self.subreddit_list:
                        if body != '': #empty
                            if body[0] != '[' and not 'http' in body and not 'www' in body and len(body) > 3:
                                body = body.replace('\n',' ').replace('\r',' ').replace('\t',' ')\
                                    .replace('&lt;','<').replace('&gt;','>').replace('&amp;','&') #some modifications
                                with open(os.path.join(output_dir,subreddit + '.txt'),'a') as subreddit_file:
                                    subreddit_file.write(body + '\n')

                    #statistics
                    if self.debug_mode:
                        hist[subreddit] = hist.get(subreddit, 0) + 1

                except:
                    logging.info("[%0d][%s] is not a legal json - skipping"%(count,line))

                    #statistics
                    if self.debug_mode:
                        hist['ERROR'] = hist.get('ERROR', 0) + 1

                if self.debug_mode and count % 10000 == 0:
                    hist['TOTAL_COMMENTS'] = count
                    with open(hist_path, 'w') as f:
                        json.dump(hist,f,ensure_ascii=False, encoding='utf8')

                count += 1

    def summary(self,output_dir):

        hist_path = os.path.join(output_dir, 'hist.json')

        for file in os.listdir(output_dir):
            if file.endswith('.txt'):
                subreddit = file.split('.')[0]
                with open(os.path.join(output_dir,file), "r") as f:
                    comments = sum(bl.count("\n") for bl in self.text_block(f))
                logging.info("[%0s] contains [%0d] comments"%(subreddit,comments))

        # if self.debug_mode:
        #     with open(hist_path, 'r') as f:
        #         hist = json.load(f)
        #
        #     #sort
        #     hist_sort = sorted(hist.items(), key=operator.itemgetter(1))
        #     hist_sort = list(reversed(hist_sort))
        #
        #     top = 300
        #     logging.info("[[TOP %0d]]"%top)
        #     for i in range(top):
        #         logging.info(str(hist_sort[i]))

    def merge_subreddits(self):

        dirs = [os.path.join(self.output_dir,f)
                for f in os.listdir(self.output_dir)
                if os.path.isdir(os.path.join(self.output_dir,f)) and f != 'log']

        #merge
        for subreddit in self.subreddit_list:
            source = [os.path.join(dir, subreddit + '.txt')
                      for dir in dirs
                      if os.path.exists(os.path.join(dir, subreddit + '.txt'))]
            target = os.path.join(self.output_dir, subreddit + '.txt')
            self.merge_shuffle_txt(source,target)


        # #done - removing dirs
        # for dir in dirs:
        #     shutil.rmtree(dir)


    def process_single_file(self,bz2):

        tmp_json = bz2 + '.json'
        tmp_output = os.path.join(self.output_dir,os.path.basename(bz2).replace('.bz2',''))

        if os.path.exists(tmp_output):
            shutil.rmtree(tmp_output)
        os.mkdir(tmp_output)

        logging.info("extructing [%0s]..."%bz2)
        self.extract_bz2_file(bz2,tmp_json)
        logging.info("processing [%0s]..." % bz2)
        self.json_process(tmp_json,tmp_output)
        logging.info("deleting temp file [%0s]..." % bz2)
        os.remove(tmp_json)
        logging.info("[SUMMARY][%0s]"%bz2)
        self.summary(tmp_output)

    def prepare_data(self):

        super(Reddit_data_handler, self).prepare_data()

        self.logger_announce('start handling reddit data')

        #find all input files and prepare a list
        bz2_files = []
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if '.bz2' in file:
                    bz2_files.append(os.path.join(root,file))
        bz2_files.sort(key=lambda x: x.lower())

        # skip existing subreddits
        existing_subreddits = [s.split('.')[0] for s in os.listdir(self.output_dir) if s.endswith('.txt')]
        logging.info("%0s.txt already exists -> skipping" % existing_subreddits)
        self.subreddit_list = [e for e in self.subreddit_list if e not in existing_subreddits]

        # thread parameters
        num_bz2_files = len(bz2_files)
        threads = []
        threads_batch_size = 5
        thread_last_batch_size = num_bz2_files % threads_batch_size
        thread_batch_num = num_bz2_files // threads_batch_size

        for i in range(thread_batch_num):

            # prepare threads
            for j in range(threads_batch_size):
                thread = threading.Thread(target=self.process_single_file, args=(bz2_files[threads_batch_size * i + j],))
                threads.append(thread)

            # run threads
            for thread in threads:
                thread.start()

            # wait for done
            for thread in threads:
                thread.join()

            # remove threads
            threads = []

        # handle last batch
        for j in range(thread_last_batch_size):
            thread = threading.Thread(target=self.process_single_file, args=(bz2_files[threads_batch_size * thread_batch_num + j],))
            threads.append(thread)

        # run threads
        for thread in threads:
            thread.start()

        # wait for done
        for thread in threads:
            thread.join()

        # remove threads
        threads = []

        logging.info("[ALL DONE - MERGING]")
        self.merge_subreddits()

        self.logger_announce('done handling reddit data')

"""
H5_data_handler
"""

class H5_data_handler(Data_handler):

    def __init__(self,name,label2files_dict,output_dir,seq_len=128,base_element='char',debug_mode=False):

        super(H5_data_handler, self).__init__(output_dir)
        self.base_element = base_element # supporting {char,word}
        self.label2files_dict = label2files_dict
        self.debug_mode = debug_mode
        self.seq_len = seq_len
        self.name = name
        self.dataset_name = self.name + '_seq-' + str(seq_len) + '_' + self.base_element + '-based' + '_classes-' +str(len(label2files_dict.keys()))

        # costs
        self.max_rows_in_memory = 1 * 1024 * 1024
        self.num_labels = len(self.label2files_dict)

        # label dicts
        self.label2num_dict = {label: i for i, label in enumerate(self.label2files_dict.keys())}
        self.num2label_dict = {i: label for i, label in enumerate(self.label2files_dict.keys())}


    def lines2tags(self,lines,label):

        tags = np.ones([len(lines),self.seq_len],dtype=self.data_type) * self.tag_dict['END_TAG']
        y = np.ones([len(lines),1],dtype=self.data_type) * self.label2num_dict[label]
        dict_keys = list(self.tag_dict.keys())
        unk_tag = self.tag_dict['UNK']

        for i, line in enumerate(lines):
            
            if self.base_element == 'word':
                _line = lines[i].split(' ')
            else:
                _line = copy(line)
                
            for j in range(min(self.seq_len,len(_line))):
                tags[i, j] = self.tag_dict.get(_line[j],unk_tag)

        return tags , y

    def h5_create(self,num_lines,h5_path):

        logging.info("creating [%0s]..." % h5_path)

        with h5py.File(h5_path, 'w') as h5:
            h5_tags = h5.create_dataset('tags',shape=(num_lines,self.seq_len),
                                             dtype=self.data_type, compression="gzip")
            h5_labels = h5.create_dataset('labels',shape=(num_lines,1),
                                               dtype=self.data_type, compression="gzip")

            h5_labels[...] = -1 * np.ones(shape=h5_labels.shape, dtype=self.data_type) # inserting invalid label for safety

    def h5_append(self,tags,labels,iter,num_label,rows_per_label,h5_path):

        label_offset = num_label * rows_per_label
        start_idx = label_offset + self.max_rows_in_memory * iter
        end_idx = start_idx + labels.shape[0]

        with h5py.File(h5_path, 'a') as h5:
            h5['tags'][start_idx:end_idx,:] = tags
            h5['labels'][start_idx:end_idx, :] = labels

    def h5_shuffle(self,h5_path):

        logging.info("shuffling [%0s]..."%h5_path)
        t0 = time.time()

        with h5py.File(h5_path, 'r+') as h5_orig :#, h5py.File(h5_temp_path, 'w') as h5_temp:

            #load all dataset to memory - a bad habit but the fastes way to do this
            tags = np.array(h5_orig['tags'])
            labels = np.array(h5_orig['labels'])

            # and shuffle
            permut = np.random.permutation(h5_orig['labels'].shape[0])
            h5_orig['tags'][...] = tags[permut,:]
            h5_orig['labels'][...] = labels[permut, :]

        logging.info("shuffling took [%0.2f SEC]" %(time.time() - t0))

    def json_dump(self,dict,json_path):
        with open(json_path, 'a',encoding='utf8') as f:
            json.dump(dict, f, ensure_ascii=False)
            f.write('\n')

    def create_tag_dict(self,file_list,rows_per_file):

        lines = []
        for file in file_list:
            with open(file,'r',encoding='utf8') as f:
                lines += [f.readline().replace('\n', '').replace('\t', '').replace('\r', '') for _ in range(rows_per_file)]

        if self.base_element == 'char':
            counts = collections.Counter(char for line in lines for char in line)
        elif self.base_element == 'word':
            counts = collections.Counter(word for line in lines for word in line.split(' '))

        tag_dict = {'UNK': 0, 'END_TAG': 1}
        cur_tag = len(tag_dict.keys())

        for char, count in counts.most_common(50000):
            if char not in tag_dict and count > 100:
                tag_dict[char] = cur_tag
                cur_tag += 1

        dict_len = len(tag_dict.keys())

        # choose the right data type
        if dict_len <= 2 ** 8:
            self.data_type = np.uint8
            self.tag_size = 1 #[Bytes]
        elif dict_len <= 2 ** 16:
            self.data_type = np.uint16
            self.tag_size = 2  # [Bytes]
        elif dict_len <= 2 ** 32:
            self.data_type = np.uint32
            self.tag_size = 4  # [Bytes]
        else:
            self.data_type = np.uint64
            self.tag_size = 8  # [Bytes]

        return tag_dict

    def prepare_data(self):

        super(H5_data_handler, self).prepare_data()

        self.logger_announce('start handling h5 process')

        logging.info("[CREATING DATASET][%0s]"%self.dataset_name)
        logging.info(str(self.label2files_dict))

        h5_path = os.path.join(self.output_dir, self.dataset_name + '.h5')
        json_path = os.path.join(self.output_dir, self.dataset_name + '.json')

        # merge files with same label
        merge_dir = os.path.join(self.output_dir,'merge')
        if not os.path.isdir(merge_dir):
            os.mkdir(merge_dir)

        # merge and shuffle text files from the same tags
        label2merge_dict = {label: os.path.join(merge_dir,label + '.txt') for label in self.label2files_dict.keys()}
        for label in self.label2files_dict.keys():
            if len(self.label2files_dict[label]) ==1: # no merge needed
                label2merge_dict[label] = self.label2files_dict[label][0]
            elif os.path.exists(label2merge_dict[label]):
                pass
            else: # merge needed
                self.merge_shuffle_txt(source_list=self.label2files_dict[label],
                                       target=label2merge_dict[label])

        # choose num of rows per label according to the shortest file
        logging.info('choosing num of rows per label..')
        label2rows_dict = {}
        for label in label2merge_dict.keys():
            with open(label2merge_dict[label], 'r',encoding='utf8') as f:
                comments = sum(bl.count("\n") for bl in self.text_block(f))
                label2rows_dict[label] = comments

        rows_per_label = min(label2rows_dict.values())

        logging.info("num of raws per label:\n%0s"%str(label2rows_dict))
        logging.info("rows_per_label is the minimum [%0d] rows"%rows_per_label)

        # create char2tag dict
        logging.info("creating tag dict...")
        self.tag_dict = self.create_tag_dict(file_list = label2merge_dict.values(), rows_per_file = rows_per_label)
        self.inv_tag_dict = {self.tag_dict[key] : key for key in self.tag_dict.keys()} # inv dict
        logging.info("created tag dict with len [%0d], using [%0s] to encode tags"%(len(self.tag_dict),str(self.data_type)))
        logging.info("40 first tokens in dict:")
        logging.info(str(list(self.tag_dict.keys())[:40]))

        # split to iteration - preventing memory overload
        num_iters = rows_per_label // self.max_rows_in_memory
        last_iter_size = rows_per_label % self.max_rows_in_memory

        # processing the merged files to h5
        self.h5_create(num_lines= rows_per_label * self.num_labels, h5_path=h5_path)

        for num_label, label in enumerate(label2merge_dict.keys()):
            logging.info("[%0s] start processing label"%label)
            logging.info("[%0s][%0s] start processing file" % (label, label2merge_dict[label]))


            with open(label2merge_dict[label], 'r',encoding='utf8') as file:

                for iter in tqdm(range(num_iters+1)):

                    num_lines = self.max_rows_in_memory
                    if iter == num_iters:
                        num_lines = last_iter_size

                    lines = [file.readline().replace('\n','').replace('\t','').replace('\r','') for _ in range(num_lines)]
                    tags, y = self.lines2tags(lines, label)
                    self.h5_append(tags,y,iter,num_label,rows_per_label,h5_path)

        # save and shuffle output
        self.h5_shuffle(h5_path)
        if os.path.exists(json_path):
            os.remove(json_path)
        self.json_dump(self.num2label_dict,json_path=json_path)
        self.json_dump(self.tag_dict, json_path=json_path)

        #check output
        self.h5_sanity_check(h5_path=h5_path,json_path=json_path)

        self.logger_announce('done handling h5 process')

    def h5_sanity_check(self,h5_path,json_path):

        logging.info("[%0s] start h5_sanity_check"%h5_path)

        with open(json_path, 'r',encoding='utf8') as f:
            num2label = json.loads(f.readline().replace('\n',''))
            readed_tag_dict = json.loads(f.readline().replace('\n',''))
            readed_tag_dict = {key: int(readed_tag_dict[key]) for key in readed_tag_dict.keys()}
            readed_inv_tag_dict = {readed_tag_dict[key]: key for key in readed_tag_dict.keys()}


        with h5py.File(h5_path, 'r') as h5:

            for key in h5.keys():
                logging.info("[%0s] %0s"%(key,str(h5[key].shape)))

            logging.info("some samples:")

            for line in range(100,130):
                #extract sentance
                sent = ''
                i = 0
                while i < self.seq_len:
                    if h5['tags'][line,i] != readed_tag_dict['END_TAG'] :
                        sent += readed_inv_tag_dict[int(h5['tags'][line,i])]
                        if self.base_element == 'word':
                            sent += ' '
                    else:
                        break
                    i += 1
                logging.info(num2label[str(h5['labels'][line][0])] + '\t\t' + sent)

            labels = h5['labels']
            tags = h5['tags']

            # # spent some to time to try and load all the data at once
            # if self.debug_mode:
            #     t0 = time.time()
            #     labels = np.array(labels)
            #     tags = np.array(tags)
            #     logging.info("loading all data took [%0.1f SEC]"%(time.time() - t0))

            logging.info("data size in memory is [%0.2f MB]"%(self.tag_size * tags.shape[0] * (tags.shape[1] + labels.shape[1]) / (1024. * 1024.)))
            logging.info('labels histogram:')
            unique, counts = np.unique(labels, return_counts=True)
            logging.info(str(dict(zip(unique, counts))))



def sandbox():
    pass

if __name__ == '__main__':
    desc = "DATA PREPROCESS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('mode', type=str, choices=['reddit_parse', 'reddit', 'news', 'news_en_only','sanity','short'],
                        help='supported modes are {reddit_parse,reddit_h5,news_h5,news_h5_eng_only,sanity,short}')
    parser.add_argument('--base_element', type=str, default= 'char', choices=['char','word'],
                        help='supported {char,word}')
    FLAGS = parser.parse_args()

    if FLAGS.mode == 'reddit_parse':

        subreddits =     ['nfl','nba','gaming','soccer','movies','relationships','anime',
         'electronic_cigarette','Fitness','technology','pokemon','PokemonPlaza',
         'FIFA','Android','OkCupid','halo','bodybuilding','food','legaladvice',
         'skyrim','formula1','DnD','Guitar','Homebrewing','DIY','relationship_advice',
         'StarWars']

        R = Reddit_data_handler(input_dir='/Volumes/###/reddit-dataset/reddit_data',
                                output_dir='/Volumes/###/reddit-dataset/reddit_out',
                                subreddit_list=subreddits,
                                debug_mode=True)

        R.prepare_data()

        exit(0)

    elif FLAGS.mode == 'reddit':
        label2files = {'relationships':
                            ['/Volumes/###/reddit/reddit_out/relationships.txt'],
                       'nba':
                            ['/Volumes/###/reddit/reddit_out/nba.txt'],
                       'fitness':
                           ['/Volumes/###/reddit/reddit_out/Fitness.txt'],
                       'android':
                           ['/Volumes/###/reddit/reddit_out/Android.txt'],
                       'electronic_cigarette':
                           ['/Volumes/###/reddit/reddit_out/electronic_cigarette.txt'],
                       }
        out_dir = '/Volumes/###/reddit/reddit_h5'

    elif FLAGS.mode == 'news':

        label2files = {'en_news':
                       [
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2007.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2008.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2009.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2010.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2011.en.shuffled'],
                       'es_news':
                       [
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2007.es.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2008.es.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2009.es.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2010.es.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2011.es.shuffled'],
                       'de_news':
                       [
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2007.de.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2008.de.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2009.de.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2010.de.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2011.de.shuffled'],
                       'cs_news':
                            [
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2007.cs.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2008.cs.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2009.cs.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2010.cs.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2011.cs.shuffled'],
                       'fr_news':
                           [
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2007.fr.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2008.fr.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2009.fr.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2010.fr.shuffled',
                            '/Volumes/###/news/1-billion-word-language-modeling/news.2011.fr.shuffled'],
                       }

        out_dir = '/Volumes/###/news/news_h5'

    elif FLAGS.mode == 'news_en_only':
        label2files = {'en_news':
                       ['/Volumes/###/news/1-billion-word-language-modeling/europarl-v6.en',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2007.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2008.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2009.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2010.en.shuffled',
                        '/Volumes/###/news/1-billion-word-language-modeling/news.2011.en.shuffled']
                       }
        out_dir = '/Volumes/###/news/news_en_only_h5'

    elif FLAGS.mode == 'short':
        label2files = {'en_news':
                       ['/Volumes/###/news/1-billion-word-language-modeling/europarl-v6.en']
                       }
        out_dir = '/Volumes/###/news/short_h5'

    elif FLAGS.mode == 'sanity':
        label2files = {'en_news':
                       ['/Volumes/###/news/1-billion-word-language-modeling/sanity.en']
                       }
        out_dir = '/Volumes/###/news/sanity_h5'
    else:
        raise NotImplementedError()

    H5 = H5_data_handler(name=FLAGS.mode,label2files_dict=label2files,
                         output_dir=out_dir,
                         seq_len=32,
                         base_element=FLAGS.base_element,
                         debug_mode=False)
    H5.prepare_data()




