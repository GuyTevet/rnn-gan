import os
import numpy as np
import datetime
import h5py
from copy import copy
import time
import random
import json

#local
import data

"""
Runtime_data_handler
"""

class Runtime_data_handler(object):

    def __init__(self,h5_path,json_path,batch_size=64,seq_len=64,teacher_helping_mode='th_extended',use_labels=True,use_var_len=True):

        self.h5_path = h5_path
        self.json_path = json_path
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.output_seq_len = seq_len                       # fixed length of the tags data stracture
        self.use_var_len = use_var_len                      # use variable length for sentance in [1,seq_len] uniformply
        self.class_dict, self.tag_dict, self.inv_tag = self.load_tag_dict_from_json()

        with h5py.File(self.h5_path, 'r') as h5:
            self.h5_num_rows = h5['tags'].shape[0]
            self.h5_seq_len = h5['tags'].shape[1]

        assert self.output_seq_len <= self.h5_seq_len

        self.num_batches_per_epoch = self.h5_num_rows // self.batch_size # skipping the last, non-complete, batch

        self.h5_curr_batch_pointer = 0

    def load_tag_dict_from_json(self):
        with open(self.json_path, 'r',encoding='utf8') as f:
            class_dict = json.loads(f.readline().replace('\n',''))
            class_dict = {int(key): class_dict[key] for key in class_dict.keys()}
            tag_dict = json.loads(f.readline().replace('\n',''))
            tag_dict = {key: int(tag_dict[key]) for key in tag_dict.keys()}
            # inv_tag_dict = {tag_dict[key]: key for key in tag_dict.keys()}

            inv_tag = [None] * len(tag_dict.keys())
            for key in tag_dict.keys():
                inv_tag[int(tag_dict[key])] = key

        return class_dict, tag_dict , inv_tag

    def h5_shuffle(self,h5_path):

        print("shuffling [%0s]..."%h5_path)
        t0 = time.time()

        with h5py.File(h5_path, 'r+') as h5_orig :#, h5py.File(h5_temp_path, 'w') as h5_temp:

            #load all dataset to memory - a bad habit but the fastes way to do this
            tags = np.array(h5_orig['tags'])
            labels = np.array(h5_orig['labels'])

            # and shuffle
            permut = np.random.permutation(h5_orig['labels'].shape[0])
            h5_orig['tags'][...] = tags[permut,:]
            h5_orig['labels'][...] = labels[permut, :]

        print("shuffling took [%0.2f SEC]" %(time.time() - t0))

    def epoch_start(self,start_batch_id = 0,seq_len=64):
        self.output_seq_len = seq_len
        self.h5_curr_batch_pointer = start_batch_id * self.batch_size
        self.h5_shuffle(self.h5_path)

    def epoch_end(self):
        # self.h5_shuffle(self.h5_path)
        pass

    def get_num_batches_per_epoch(self):
        return self.num_batches_per_epoch

    def get_batch(self,create_mask=False):

        if self.h5_curr_batch_pointer+self.batch_size >= self.h5_num_rows:
            print("data ended. shuffling...")
            self.h5_shuffle(self.h5_path)

        with h5py.File(self.h5_path, 'r') as h5:
            tags = np.array(h5['tags'][self.h5_curr_batch_pointer:self.h5_curr_batch_pointer+self.batch_size,:self.output_seq_len])
            labels = np.squeeze(h5['labels'][self.h5_curr_batch_pointer:self.h5_curr_batch_pointer+self.batch_size])

        # print("batch [%0d : %0d]"%(self.h5_curr_batch_pointer,self.h5_curr_batch_pointer+self.batch_size)) # for debug

        # # process
        # tags = self.process_tags(tags,create_mask=create_mask)

        # increment batch pointer
        self.h5_curr_batch_pointer += self.batch_size


        return tags, labels


    # def process_tags(self,tags,create_mask=False):
    #
    #     feed_tags = self.tag_dict['END_TAG'] * np.ones(shape=[tags.shape[0],self.output_seq_len],dtype=np.uint8) #copy(tags)[:,:self.output_seq_len]
    #
    #     # cut sample in var lens
    #     len_list = []
    #     for sample_i in range(feed_tags.shape[0]):
    #         if self.use_var_len == True:
    #             len = random.randrange(1, self.output_seq_len + 1)
    #         else:
    #             len = self.output_seq_len
    #         feed_tags[sample_i,:len] = tags[sample_i,:len]
    #         len_list.append(len)
    #
    #
    #
    #     return feed_tags

        # #create mask if needed
        # mask = np.zeros(shape=[tags.shape[0],self.output_seq_len],dtype=np.bool)
        # if create_mask:
        #     for i, length in enumerate(len_list):
        #         if self.teacher_helping_mode == 'th_legacy':
        #             window_size = 1
        #             window_offset = length - 1
        #         elif self.teacher_helping_mode == 'th_extended':
        #             window_size = random.randrange(1, length + 1)
        #             window_offset = random.randrange(0, length - window_size + 1)
        #         elif self.teacher_helping_mode == 'full':
        #             window_size = length
        #             window_offset = 0
        #         else:
        #             raise TypeError('supported modes are {th_legacy,th_extended,full}')
        #
        #         mask[i,window_offset:window_offset + window_size] = 1
        #
        # start_tags = self.tag_dict['START'] * np.ones([self.batch_size,1],dtype=np.uint8)
        # end_tags = self.tag_dict['END_TAG'] * np.ones([self.batch_size, 1], dtype=np.uint8)
        # feed_tags = np.concatenate((start_tags,feed_tags,end_tags),axis=1)
        #
        # mask_zero_pad = np.zeros([self.batch_size,1],dtype=np.bool)
        # mask = np.concatenate((mask_zero_pad,mask,mask_zero_pad),axis=1)
        #
        # return feed_tags, mask



