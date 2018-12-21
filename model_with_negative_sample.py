import tensorflow as tf
import random
import pandas as pd
from utils import *
from rnn import dynamic_rnn
import datetime
import numpy as np
from Dice import dice
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy import stats

def prepare_data(filename):
    with open(filename) as f:
        uid_list=[]
        uid_age_list=[]
        uid_gender_list=[]
        mid_list=[]
        label_list=[]
        item_his=[]
        cat_his=[]
        brand_his=[]
        his_time=[]
        act_his=[]
        for cnt,line in enumerate(f):
            if cnt ==0:
                continue
            ar=line.strip('\n').split(',')
            uid_list.append(ar[0])
            uid_age_list.append(ar[1])
            uid_gender_list.append(ar[2])
            mid_list.append(ar[3])
            label_list.append(ar[4])

            his=ar[5].split('#')
            item_list=[]
            cat_list=[]
            brand_list=[]
            time_list=[]
            act_list=[]
            if his != ['']:
                for act in his:
                    ac=act.split(':')
                    item_list.append(ac[0])
                    cat_list.append(ac[1])
                    brand_list.append(ac[2])
                    time_list.append(ac[3])
                    act_list.append(ac[4])
            item_his.append(item_list)
            cat_his.append(cat_list)
            brand_his.append(brand_list)
            his_time.append(time_list)
            act_his.append(act_list)

    return uid_list,uid_age_list,uid_gender_list,mid_list,label_list,item_his,cat_his,brand_his,his_time,act_his

def build_dict_of_user_profile():
    log=pd.read_csv('/mydata/data_format1/data/user_log_format1.csv')
    log.time_stamp=log.time_stamp.apply(lambda x:(datetime.datetime.strptime(str(x),'%m%d')-datetime.datetime(1900,1,1,0,0)).days)
    log=pd.get_dummies(log,columns=['action_type'])
    log=log.sort_values(['user_id','time_stamp'])
    log_d=log.drop(['user_id','cat_id','brand_id','seller_id'],axis=1)
    item_list=list(log_d.item_id.drop_duplicates())
    ns_l=[]
    for item in log_d.item_id:
        ns=np.random.choice(item_list)
        while ns==item:
            ns=np.random.choice(item_list)
        ns_l.append(ns)
    noclk_item=pd.DataFrame(ns_l)
    noclk_item.columns=['noclk_item']
    log_ns=pd.concat([log_d,noclk_item],axis=1)
    log_ar=np.array(log_ns)
    grp=log.groupby('user_id').size()
    uid_fe_dc={}
    position=0
    for uid in grp.index:
        uid_profile=log_ar[position:position+grp.ix[uid]]
        uid_fe_dc[uid]=uid_profile
    return uid_fe_dc

def build_dict_of_user_info():
    user_info=pd.read_csv('/mydata/data_format1/user_info.csv')
    user_info_ar=np.array(user_info.drop('user_id',axis=1))
    uid_info_dc={}
    for uid in user_info.user_id:
        uid_info_dc[uid]=user_info_ar[user_info.user_id==uid]
    return uid_info_dc

def build_dict_of_item():
    log=pd.read_csv('/mydata/data_format1/data/user_log_format1.csv')
    log=log[['item_id','cat_id','brand_id','seller_id']]
    log_i=log.drop_duplicates('item_id')
    log_i.set_index('item_id',inplace=True)
    item_info_dc={}
    for item in log_i.index:
        item_info_dc[item]=np.array(log_i.ix[item])
    return item_info_dc


item_info_dc=build_dict_of_item()
uid_info_dc=build_dict_of_user_info()
uid_fe_dc=build_dict_of_user_profile()

train=pd.read_csv('/mydata/data_format1/data/train_format1.csv')
train=pd.get_dummies(train,columns=['label'])
print('finish prepare!!!')

def get_batch(batch_size,train_batch):
    uid_ph=np.array([i for i in train_batch.user_id])
    uid_info_ph=np.array([uid_info_dc[i] for i in train_batch.user_id]).reshape([batch_size,-1])
    mid_ph=np.array([i for i in train_batch.merchant_id])
    len_ph=np.array([uid_fe_dc[i].shape[0] for i in train_batch.user_id])
    max_len=max(len_ph)
    fe_ph=np.array([ np.concatenate([uid_fe_dc[i],np.zeros([max_len-uid_fe_dc[i].shape[0],uid_fe_dc[i].shape[1]])-1],0)   for i in train_batch.user_id])
    click_item_ph=map(lambda x: item_info_dc[x],list(fe_ph[:,:,0]))
    noclick_item_ph=map(lambda x:item_info_dc[x],list(fe_ph[:,:,-1]))
    label_ph=np.array(train_batch[['label_0','label_1']])
    mask_ph=np.zeros([batch_size,max_len])
    for i,length in enumerate(len_ph):
        mask_ph[i,:length]=1 
    return uid_ph,uid_info_ph,mid_ph,len_ph,click_item_ph[:,:,0],click_item_ph[:,:,2],click_item_ph[:,:,1],fe_ph[:,:,0],fe_ph[:,:,1:],label_ph,mask_ph,
    noclick_item_ph[:,:,0],noclick_item_ph[:,:,2],noclick_item_ph[:,:,1]
    
def evaluate(batch_size):
    test=data.ix[user_list[:20000]]
    test=test.reset_index()
    y_prob=[]
    for i in tqdm(range(test.shape[0]//batch_size)):
        test_batch=test.ix[i*batch_size:(i+1)*batch_size-1]
        uid_ph,uid_info_ph,mid_ph,len_ph,cat_his_ph,mid_his_ph,brand_his_ph,his_time_ph,act_his_ph,label_ph,mask_ph,noclk_cat_his_ph,noclk_mid_his_ph,noclk_brand_his_ph=get_batch(test_batch.shape[0],test_batch)
        _y=sess.run(prob,feed_dict={
                            uid:uid_ph,
                            uid_info:uid_info_ph,
                            mid:mid_ph,
                            seq_len:len_ph,
                            cat_his:cat_his_ph,
                            mid_his:mid_his_ph,
                            brand_his:brand_his_ph,
                            his_time:his_time_ph,
                            act_his:act_his_ph,
                            mask:mask_ph})
        y_prob.append(_y)
    y_prob=np.array(y_prob).reshape(-1,2)
    if (test.shape[0]//batch_size)*batch_size<test.shape[0]:
        test_batch=test.ix[(i+1)*batch_size:]
        uid_ph,uid_info_ph,mid_ph,len_ph,cat_his_ph,mid_his_ph,brand_his_ph,his_time_ph,act_his_ph,label_ph,mask_ph,noclk_cat_his_ph,noclk_mid_his_ph,noclk_brand_his_ph=get_batch(test_batch.shape[0],test_batch)
        _y=sess.run(prob,feed_dict={
                            uid:uid_ph,
                            uid_info:uid_info_ph,
                            mid:mid_ph,
                            seq_len:len_ph,
                            cat_his:cat_his_ph,
                            mid_his:mid_his_ph,
                            brand_his:brand_his_ph,
                            his_time:his_time_ph,
                            act_his:act_his_ph,
                            mask:mask_ph})
        y_prob=np.concatenate([y_prob,_y],0)
    return y_prob,test



    def auxiliary_loss(h_states, click_seq, noclick_seq, mask, stag = None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = auxiliary_net(click_input_, stag = stag)[:, :, 0]
        noclick_prop_ = auxiliary_net(noclick_input_, stag = stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

'''
##################################################################
                          
                          build graph

##################################################################
'''

with tf.name_scope('Inputs'):
    uid=tf.placeholder(tf.int32,[None,],name='uid')
    uid_info=tf.placeholder(tf.float32,[None,12],name='uid_info')
    mid=tf.placeholder(tf.int32,[None,],name='mid')
    mid_his=tf.placeholder(tf.int32,[None,None],name='mid_his')
    label=tf.placeholder(tf.float32,[None,2],name='label')
    cat_his=tf.placeholder(tf.int32,[None,None],name='cat_his')
    brand_his=tf.placeholder(tf.int32,[None,None],name='brand_his')
    his_time=tf.placeholder(tf.float32,[None,None],name='time_his')
    act_his=tf.placeholder(tf.float32,[None,None,4],name='act_his')
    seq_len=tf.placeholder(tf.float32,[None,],name='Seq_len')
    mask=tf.placeholder(tf.float32,[None,None],name='mask')
    lr=tf.placeholder(tf.float32,name='learning_rate')
    noclk_cat_his=tf.placeholder(tf.int32,[None,None],name='noclk_cat_his')
    noclk_brand_his=tf.placeholder(tf.int32,[None,None],name='noclk_brand_his')
    noclk_mid_his=tf.placeholder(tf.int32,[None,None],name='noclk_mid_his')

with tf.name_scope('Embedding_Layer'):
    em_mat_mid=tf.get_variable('Embedding_Mid',[4996,64])
    mid_em=tf.nn.embedding_lookup(em_mat_mid,mid)
    mid_his_em=tf.nn.embedding_lookup(em_mat_mid,mid_his)
    noclk_mid_his_em=tf.nn.embedding_lookup(em_mat_mid,noclk_mid_his)

    em_mat_cat=tf.get_variable('Embedding_Cat',[1659,32])
    cat_his_em=tf.nn.embedding_lookup(em_mat_cat,cat_his)
    noclk_cat_his_em=tf.nn.embedding_lookup(em_mat_cat,noclk_cat_his)

    em_mat_brand=tf.get_variable('Embedding_Brand',[8500,96])
    brand_his_em=tf.nn.embedding_lookup(em_mat_brand,brand_his)
    noclk_brand_his_em=tf.nn.embedding_lookup(em_mat_brand,noclk_brand_his)

with tf.name_scope('rnn1'):
    his=tf.concat([cat_his_em,brand_his_em,tf.expand_dims(his_time,-1),act_his,mid_his_em],2)
    mid_his_sum=tf.reduce_sum(mid_his_em,1)
    rnnouts1,rnnfinalouts1=tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(128),inputs=his,sequence_length=seq_len,dtype=tf.float32)

with tf.name_scope('Aux_loss'):
    noclk_his=tf.concat([noclk_cat_his_em,noclk_brand_his_em,tf.expand_dims(his_time,-1),act_his,noclk_mid_his_em],2)
    aux_loss=auxiliary_loss(rnnouts1,his,noclk_his,mask)

with tf.name_scope('Attention_layer'):
    att_outputs,alphas=din_fcn_attention(mid_em,rnnouts1,100,mask,mode='list',return_alphas=True)


with tf.name_scope('rnn2'):
    rnnouts2,rnnfinalouts2=dynamic_rnn(VecAttGRUCell(128),inputs=rnnouts1,att_scores=tf.expand_dims(alphas, -1),sequence_length=seq_len,dtype=tf.float32)
#    rnnouts2,rnnfinalouts2=tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(1024),inputs=rnnouts1,sequence_length=seq_len,dtype=tf.float32)


with tf.name_scope('Dense'):
    uid_feature=tf.concat([uid_info,rnnfinalouts2,mid_em],axis=1)
    W_h1=tf.get_variable('W_h1',[uid_feature.shape[1],256])
    b_h1=tf.get_variable('b_h1',[256,])
    h1=tf.nn.xw_plus_b(uid_feature,W_h1,b_h1)
    h1=dice(h1,name='dice_1')
    W_h2=tf.get_variable('W_h2',[256,64])
    b_h2=tf.get_variable('b_h2',[64,])
    h2=tf.nn.xw_plus_b(h1,W_h2,b_h2)
    h2=dice(h2,name='dice_2')
    W_h3=tf.get_variable('W_h3',[64,2])
    b_h3=tf.get_variable('b_h3',[2,])
    h3=tf.nn.xw_plus_b(h2,W_h3,b_h3)
    prob=tf.nn.softmax(h3)

with tf.name_scope('Metrics'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h3,labels=label))
    total_loss=loss+aux_loss
    optimizer=tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=50)

learning_rate=0.01
data=train.set_index('user_id')
user_list=list(set(data.index))
train=data.ix[user_list[20000:]]
train=train.reset_index()
test=data.ix[user_list[:20000]]
test=test.reset_index()
for cnt,epoch in enumerate(range(10)):
    loss_tt=[]
    for it in range(train.shape[0]//128):
        train_batch=train.ix[it*128:(it+1)*128-1]
        uid_ph,uid_info_ph,mid_ph,len_ph,cat_his_ph,mid_his_ph,brand_his_ph,his_time_ph,act_his_ph,label_ph,mask_ph,noclk_cat_his_ph,noclk_mid_his_ph,noclk_brand_his_ph=get_batch(train_batch.shape[0],train_batch)
        _,loss_tr,_y=sess.run([optimizer,loss,prob],feed_dict={uid:uid_ph,
                                        uid_info:uid_info_ph,
                                        mid:mid_ph,
                                        seq_len:len_ph,
                                        cat_his:cat_his_ph,
                                        mid_his:mid_his_ph,
                                        brand_his:brand_his_ph,
                                        his_time:his_time_ph,
                                        act_his:act_his_ph,
                                        label:label_ph,
                                        mask:mask_ph,
                                        lr:learning_rate,
                                        noclk_brand_his:noclk_brand_his_ph,
                                        noclk_cat_his:noclk_brand_his_ph,
                                        noclk_mid_his:noclk_mid_his_ph})
        
        if it%1000==0:
            saver.save(sess,'./model_with_negative_sample/model',global_step=(epoch*train.shape[0]//128+it))
        print('Epoch:%s'%(epoch))
        print('Iter :%s'%(it))
        print('Loss :%s'%(loss_tr))
        try:
            print('Auc  :%s'%(roc_auc_score(label_ph[:,1],_y[:,1])))
        except ValueError as e:
            print(e)
    if (train.shape[0]//128)*128<train.shape[0]:
        train_batch=train.ix[(it+1)*128:]
        uid_ph,uid_info_ph,mid_ph,len_ph,cat_his_ph,mid_his_ph,brand_his_ph,his_time_ph,act_his_ph,label_ph,mask_ph,noclk_cat_his_ph,noclk_mid_his_ph,noclk_brand_his_ph=get_batch(train_batch.shape[0],train_batch)
        _,loss_tr,_y=sess.run([optimizer,loss,prob],feed_dict={uid:uid_ph,
                                        uid_info:uid_info_ph,
                                        mid:mid_ph,
                                        seq_len:len_ph,
                                        cat_his:cat_his_ph,
                                        mid_his:mid_his_ph,
                                        brand_his:brand_his_ph,
                                        his_time:his_time_ph,
                                        act_his:act_his_ph,
                                        label:label_ph,
                                        mask:mask_ph,
                                        lr:learning_rate,
                                        noclk_brand_his:noclk_brand_his_ph,
                                        noclk_cat_his:noclk_brand_his_ph,
                                        noclk_mid_his:noclk_mid_his_ph})
        loss_tt.append(loss_tr)
        if it%1000==0:
            saver.save(sess,'./model_with_negative_sample/model',global_step=(epoch*train.shape[0]//128+it))
        print('Epoch:%s'%(epoch))
        print('Iter :%s'%(it))
        print('Loss :%s'%(loss_tr))
        print('Losst:%s'%(sum(loss_tt)/len(loss_tt)))
        try:
            print('Auc  :%s'%(roc_auc_score(label_ph[:,1],_y[:,1])))
        except ValueError as e:
            print(e)
    y_prob,test=evaluate(128)
    print('AUC  :%s'%(roc_auc_score(test.label_1,y_prob[:,1])))
    if cnt==4:
        learning_rate*=0.5


saver.save(sess,'./model_with_negative_sample/model',global_step=(epoch*train.shape[0]//128+it))
