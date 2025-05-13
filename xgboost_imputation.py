#!/usr/bin python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np
import pandas as pd
import datetime
import time
import pickle
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
from utils import *
import math
import collections
import re

def load_data(args):

    #loading the files has been based on deep-hla code
    logger.log('Training processes started at {}.'.format(time.ctime()))
    
    if args.build == 'grch38':
        hla_info = pd.DataFrame({"HLA_F": {"pos": "29724053"},"HLA_G": {"pos": "29828432"},"HLA_A": {"pos": "29943150"},"HLA_E": {"pos": "30490120"},"HLA_C": {"pos": "31271470"},"HLA_B": {"pos": "31356562"},"HLA_DRA": {"pos": "32442957"},"HLA_DRB3": {"pos": "32453610"},"HLA_DRB5": {"pos": "32520772"},"HLA_DRB4": {"pos": "32546191"},"HLA_DRB1": {"pos": "32582967"},"HLA_DQA1": {"pos": "32641781"},"HLA_DQB1": {"pos": "32663517"},"HLA_DOB": {"pos": "32814816"},"HLA_DMB": {"pos": "32938068"},"HLA_DMA": {"pos": "32950207"},"HLA_DOA": {"pos": "33007786"},"HLA_DPA1": {"pos": "33071655"},"HLA_DPB1": {"pos": "33082951"}})
    else:
        return

    # Load files
    logger.log('Loading files...')
    logger.log('Loading Reference .bim file [' + args.ref_bim +'.bim].')
    ref_bim = pd.read_table(args.ref_bim + '.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    logger.log('Loading sample .bim file [' + args.sample + '.bim].')
    sample_bim = pd.read_table(args.sample + '.bim', sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')

    if not args.gene is None:
        start_pos = int(hla_info[args.gene]['pos']) - args.window
        end_pos = int(hla_info[args.gene]['pos']) + args.window
        logger.log('GENE '+ args.gene + 'Start position: ['+ str(start_pos)+'] - End position: ['+ str(end_pos)+']')
        sample_bim = sample_bim.query('pos > @start_pos and pos < @end_pos')
    sample_bim = sample_bim.reset_index(drop=True)
    
    #now remove non-snps (beagle encodes them strangely, and it's hard to use them)
    bim_ids = sample_bim.id.str.split(":")
    ids_tokeep = []
    for i in range(len(bim_ids)):
        if len(bim_ids[i][2]) == 1 and len(bim_ids[i][3]) == 1:
            ids_tokeep.append(i)
    sample_bim = sample_bim.iloc[ids_tokeep].reset_index(drop=True)

    
    # Extract only SNPs which exist both in reference and sample data
    concord_snp = ref_bim.pos.isin(sample_bim.pos)
    for i in range(len(concord_snp)):
        if concord_snp.iloc[i]:
            tmp = np.where(sample_bim.pos == ref_bim.iloc[i].pos)[0][0]
            if set((ref_bim.iloc[i].a1, ref_bim.iloc[i].a2)) != \
                    set((sample_bim.iloc[tmp].a1, sample_bim.iloc[tmp].a2)):
                concord_snp.iloc[i] = False
    num_concord = np.sum(concord_snp)
    logger.log('{} SNPs loaded from reference.'.format(len(ref_bim)))
    logger.log('{} SNPs loaded from sample.'.format(len(sample_bim)))
    logger.log('{} SNPs matched in position and used for training.'.format(num_concord))

    model_bim = ref_bim.iloc[np.where(concord_snp)[0]]

    model_bim.to_csv(os.path.join(args.model_dir, 'model_snps_used_' + args.gene + '.bim'), sep='\t', header=False, index=False)
    list_snps = model_bim['id']

    logger.log('Loading reference .bgl.phased file.')
    if args.use_pandas == False:
        logger.log('Reading file with open().')
        data = []
        count = 1
        num_ref = 0
        num_snps = 0
        df_hla = pd.DataFrame()
        df_snps_list = []
        df_hla_list = []
        with open(args.ref_bgl + '.bgl.phased') as my_file:
            for line in my_file:
                count=count+1
                if count % 1000 == 0:
                    logger.log('{} lines read.'.format(count))
                    data.append(line)
                    df_tmp = pd.DataFrame(data)[0].str.split(' ', expand=True)
                    if df_tmp[df_tmp[1].isin(list_snps)].shape[0] > 0:
                        df_snps_list.append(df_tmp[df_tmp[1].isin(list_snps)].iloc[:,1:])
                    if args.gene is None:
                        if df_tmp[df_tmp[1].str.startswith('HLA')].shape[0] > 0:
                            df_hla_list.append(df_tmp[df_tmp[1].str.startswith('HLA')].iloc[:,1:])
                    else:
                        if df_tmp[df_tmp[1].str.startswith(args.gene)].shape[0] > 0:
                            df_hla_list.append(df_tmp[df_tmp[1].str.startswith(args.gene)].iloc[:,1:])
                    data = []
                else:
                    data.append(line)
        df_snps = pd.concat(df_snps_list,axis=0)
        df_snps.iloc[:,df_snps.shape[1]-1] = df_snps.iloc[:,df_snps.shape[1]-1].replace(r'\n','', regex=True) 
        df_hla = pd.concat(df_hla_list,axis=0)
        df_hla.iloc[:,df_hla.shape[1]-1] = df_hla.iloc[:,df_hla.shape[1]-1].replace(r'\n','', regex=True) 
        num_ref = df_snps.shape[1] // 2  
    else:
        logger.log('Reading file with pandas.')
        ref_phased = pd.read_table(args.ref_bgl + '.bgl.phased', sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
        df_snps = ref_phased.iloc[np.where(concord_snp)[0]]
        if not args.gene is None:
            df_hla = ref_phased[ref_phased[1].str.startswith('HLA')]
        else:
            df_hla = ref_phased[ref_phased[1].str.startswith(args.gene)]
    
    logger.log('{} people loaded from reference.'.format(num_ref))
    
    return df_snps,df_hla

def remove_incomplete_haplotypes(df_snps,df_hla,args):
    df_snps_ax = df_snps.reset_index(drop=True).set_axis(range(df_snps.shape[1]),axis=1)
    df_hla_ax = df_hla.reset_index(drop=True).set_axis(range(df_hla.shape[1]),axis=1)
    chr_hla_count = np.zeros((df_hla_ax.shape[1]-1))
    for i in range(1,df_hla_ax.shape[1]):
        chr_hla_count[i-1] = df_hla_ax[i].value_counts().reindex([args.allele_present,args.allele_absent],fill_value=0)[args.allele_present]
    part_to_remove_1 = np.unique((np.where(chr_hla_count != 1)[0] )// 2)
    part_to_remove_2 = np.append(part_to_remove_1*2+1,part_to_remove_1*2+2)
    df_hla_mod = df_hla_ax.drop(part_to_remove_2,axis=1)
    df_snps_mod = df_snps_ax.drop(part_to_remove_2,axis=1)
    return df_snps_mod,df_hla_mod

def snps_transpose(df_snps,check_variance=True):
    df_tmp = df_snps.set_axis(range(df_snps.shape[1]),axis=1)
    df_tmp = df_tmp.set_index(0, drop=True)
    df_tmp = df_tmp.set_axis(range(df_tmp.shape[1]),axis=1)
    df_tmp = df_tmp.transpose()
    if check_variance:
        df_tmp = df_tmp.loc[:, (df_tmp != df_tmp.iloc[0]).any()]
    return df_tmp

def alleles_to_binary(df):
    df_colnames = pd.DataFrame({'colnames':df.columns.to_list()})['colnames'].str.split(":")
    for i in range(len(df_colnames)):
        refa = df_colnames[i][2]
        alta = df_colnames[i][3]
        df.iloc[:,i] = df.iloc[:,i].replace(refa,'0', regex=True) 
        df.iloc[:,i] = df.iloc[:,i].replace(alta,'1', regex=True)
    return df
        
def collapse_hla(df_hla,args):
    hla_coll = [None] * (df_hla.shape[1] - 1)
    df_hla_ax = df_hla.set_axis(range(df_hla.shape[1]),axis=1).reset_index(drop=True)
    for i in range(1,len(hla_coll)+1):
        hla_coll[i-1] = df_hla_ax.iloc[np.where(df_hla_ax.iloc[:,i]==args.allele_present)[0][0],0]
    return hla_coll

def min_allele_filter(_df_snps,_df_hla,args):
    if args.two_fields == True:
        _df_hla = pd.Series(_df_hla).str.extract(r'(HLA_[A-Za-z0-9_]*[0-9]*:[0-9]*)')[0].to_numpy()
    count_alleles = collections.Counter(_df_hla)
    min_ac_t = args.min_ac
    alleles_to_keep = pd.DataFrame(dict(count_alleles),index=['counts']).transpose().query('counts >= @min_ac_t').index
    indices_to_keep = sum([list(np.where(np.array(_df_hla) == j)[0]) for j in alleles_to_keep],[])
    indices_to_keep.sort()
    df_hla_filtered = np.array(_df_hla)[indices_to_keep]
    df_snps_filtered = _df_snps.reset_index(drop=True).iloc[indices_to_keep,:].reset_index(drop=True)
    logger.log('Final number of chromosomes used: ' + str(df_snps_filtered.shape[0]))
    logger.log('Final number of unique HLA alleles used: ' + str(len(np.unique(df_hla_filtered))))
    np.save(os.path.join(args.model_dir, 'list_alleles_' + args.gene + '.npy'),np.unique(df_hla_filtered))
    return df_snps_filtered,df_hla_filtered
    

def prep_for_xgboost(_snps,_hla,args):
    le = LabelEncoder()
    le.fit(_hla)
    if args.weighted_logloss == True:
        count_alleles = collections.Counter(_hla)
        allele_names = count_alleles.keys()
        count_alleles = pd.DataFrame(count_alleles, index = [0]).transpose()
        count_alleles.columns = ['counts']
        count_alleles['alleles'] = allele_names
        df_hla = pd.DataFrame({'alleles': _hla})
        df_hla = df_hla.merge(count_alleles,on='alleles',how='left',indicator=True)
        min_count = min(df_hla['counts'])
        df_hla['counts'] = min_count/df_hla['counts']
        dtrain = xgb.DMatrix(_snps.to_numpy(dtype = np.float32), label = le.transform(_hla), nthread = -1, feature_names = _snps.columns.to_list(), weight = df_hla['counts'])
    else :
        dtrain = xgb.DMatrix(_snps.to_numpy(dtype = np.float32), label = le.transform(_hla), nthread = -1, feature_names = _snps.columns.to_list())
    pickle.dump(le, open(os.path.join(args.model_dir, 'train_'+ args.gene+ '_label_encoder.pkl'), 'wb'))
    dtrain.save_binary(os.path.join(args.model_dir, 'train_'+ args.gene+ '.buffer'))
    return

def bayes_optim_hla(args):
    logger.log('Bayesian optimization at {}.'.format(time.ctime()))
    logger.log('Gene '+str(args.gene))
    dtrain = xgb.DMatrix(os.path.join(args.model_dir, 'train_'+ args.gene+ '.buffer'))
    def xgb_cv_bayes(max_depth, min_child_weight, subsample,eta,gamma,colsample_bytree,max_delta_step,n_estimators):
        if args.use_gpu == True:
            param = {'booster' : 'gbtree',
                      'max_depth' : int(max_depth),
                      'min_child_weight' : min_child_weight,
                      'eta':eta,
                      'gamma':gamma,
                      'num_class': len(set(dtrain.get_label())),
                      'subsample' : max(min(subsample,1),0), 
                      'max_delta_step' : int(max_delta_step),
                      'lambda' : 1, 
                      'alpha' : 0,
                      'objective' : 'multi:softprob',
                      'eval_metric' : 'mlogloss',
                      'nthread': args.threads,
                      'device': 'cuda',
                      'sampling_method': 'gradient_based',
                      'tree_method': 'hist',
                      'seed' : args.cv_seed}
        else:
            param = {'booster' : 'gbtree',
                      'max_depth' : int(max_depth),
                      'min_child_weight' : min_child_weight,
                      'eta':eta,
                      'gamma':gamma,
                      'num_class': len(set(dtrain.get_label())),
                      'subsample' : max(min(subsample,1),0), 
                      'max_delta_step' : int(max_delta_step),
                      'lambda' : 1, 
                      'alpha' : 0,
                      'objective' : 'multi:softprob',
                      'eval_metric' : 'mlogloss',
                      'nthread': args.threads,
                      'device': 'cpu',
                      'tree_method': 'hist',
                      'seed' : args.cv_seed}
        cv_res = xgb.cv(params = param, dtrain = dtrain, num_boost_round =int(math.ceil(n_estimators)), nfold = args.nfolds,early_stopping_rounds=10, verbose_eval = True, maximize=False)
        score_to_maximize = -1*cv_res['test-mlogloss-mean'][cv_res['test-mlogloss-mean'].argmin()]
        return score_to_maximize
    if args.use_gpu == True:
        hyperparameter_bounds = {'max_depth' : (10,100),
                              'min_child_weight' : (1,20),
                              'subsample' : (0.5,0.95),
                              'eta': (0.01,0.3),
                              'gamma' : (0.0, 1),
                              'colsample_bytree':(0.1,0.95),
                              'max_delta_step':(0,10),
                              'n_estimators': (50,300)
                              }
    else:
        hyperparameter_bounds = {'max_depth' : (10,100),
                              'min_child_weight' : (1,20),
                              'subsample' : (0.5,0.95),
                              'eta': (0.01,0.3),
                              'gamma' : (0.0, 1),
                              'colsample_bytree':(0.5,0.95),
                              'max_delta_step':(0,10),
                              'n_estimators': (50,300)
                              }
    hla_optimizer = BayesianOptimization(f=xgb_cv_bayes,pbounds = hyperparameter_bounds,verbose = 2)
    hla_optimizer.maximize(init_points = 15, n_iter = 15)
    pickle.dump(hla_optimizer.res, open(os.path.join(args.model_dir, 'train_'+ args.gene+ '_hla_optimizer.pkl'), 'wb'))
    return 
              
              
def xgb_train_hla(args):
    logger.log('XGboost training at {}.'.format(time.ctime()))
    logger.log('Gene '+str(args.gene))
    hla_optimizer = pickle.load(open(os.path.join(args.model_dir, 'train_'+ args.gene+ '_hla_optimizer.pkl'), 'rb'))
    param = pd.DataFrame(hla_optimizer).sort_values('target',ascending=False).reset_index(drop=True)['params'][0]
    n_estimators = math.ceil(param['n_estimators']*(1+1/args.nfolds))
    del param['n_estimators']
    param['nthread'] = args.threads
    param['max_depth'] = int(math.ceil(param['max_depth']))
    param['booster'] = 'gbtree'
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['tree_method'] = 'hist'
    dtrain = xgb.DMatrix(os.path.join(args.model_dir, 'train_'+ args.gene+ '.buffer'))
    param['num_class'] = len(set(dtrain.get_label()))
    xgb_res = xgb.train(params = param, num_boost_round=int(math.ceil(n_estimators)), dtrain = dtrain)
    xgb_res.save_model(os.path.join(args.model_dir, 'xgboost_model_'+ args.gene+ '.json'))
    return

def xgb_pred_hla(args):
    logger.log('Gene '+str(args.gene))
    xgb_trained_model = xgb.Booster()
    xgb_trained_model.load_model(os.path.join(args.model_dir, 'xgboost_model_'+ args.gene+ '.json'))
    return xgb_trained_model

def load_new_snps(xgb_trained_model,args):
    logger.log('Loading new snps .bgl.phased file.')
    
    #obtain the features of the trained xgboost model
    xgb_feat_names = xgb_trained_model.feature_names
    #see here later: https://stackoverflow.com/questions/42338972/valueerror-feature-names-mismatch-in-xgboost-in-the-predict-function
    
    #load the bim file of the new snps
    sample_bim = pd.read_table(args.sample_for_imputation, sep='\t|\s+', names=['chr', 'id', 'dist', 'pos', 'a1', 'a2'], header=None, engine='python')
    
    #inner join of snps in the reference and the new sample to be imputed
    list_snps = sample_bim[sample_bim['id'].isin(xgb_feat_names)].reset_index(drop=True)['id']
    
    if args.use_pandas == False:
        logger.log('Reading file with open().')
        data = []
        count = 1
        num_ref = 0
        num_snps = 0
        df_snps_list = []
        with open(args.snps_for_imputation) as my_file:
            for line in my_file:
                count=count+1
                if count % 1000 == 0:
                    logger.log('{} lines read.'.format(count))
                    data.append(line)
                    df_tmp = pd.DataFrame(data)[0].str.split(' ', expand=True)
                    if df_tmp[df_tmp[1].isin(list_snps)].shape[0] > 0:
                        df_snps_list.append(df_tmp[df_tmp[1].isin(list_snps)].iloc[:,1:])
                    data = []
                else:
                    data.append(line)
        df_snps = pd.concat(df_snps_list,axis=0)
        df_snps.iloc[:,df_snps.shape[1]-1] = df_snps.iloc[:,df_snps.shape[1]-1].replace(r'\n','', regex=True) 
        df_snps=df_snps.reset_index(drop=True)
        df_snps.columns = range(0,df_snps.shape[1])
        df_snps.rename(columns={0:'SNP'},inplace=True)
        num_sample = df_snps.shape[1] // 2  
    else:
        logger.log('Reading file with pandas.')
        ref_phased = pd.read_table(args.snps_for_imputation, sep='\t|\s+', header=None, engine='python', skiprows=5).iloc[:, 1:]
        df_snps = ref_phased[ref_phased[1].isin(list_snps)].reset_index(drop=True)
        num_sample = df_snps.shape[1] // 2  

    logger.log('{} people loaded from reference.'.format(num_sample))
    
    
    with open(args.snps_for_imputation, 'r') as file:
        first_line = re.sub('\n', " ", file.readline())
        if first_line.strip().startswith('P pedigree'):
            list_samples = re.sub('P pedigree', '', first_line)
        elif first_line.strip().startswith('I id'):
            list_samples = re.sub('I id', '', first_line)
        else:
            logger.log("ERROR: The file doesn't start with 'P pedigree' or 'I id'.")
        sys.exit(1)

    list_samples = list_samples[1:-1].split(' ')
    
    return df_snps,list_samples

def add_null_columns(df_snps,xgb_trained_model):
    current_feature_list = df_snps.iloc[:,0]
    obj_feature_list = xgb_trained_model.feature_names
    obj_feature_df = pd.DataFrame({'SNP':obj_feature_list})
    df_snps_filled = obj_feature_df.merge(df_snps,how='left',on='SNP')
    return df_snps_filled

def impute_the_hla(args,df_imputation_ready,xgb_trained_model,list_samples):
    le = pickle.load(open(os.path.join(args.model_dir, 'train_'+ args.gene+ '_label_encoder.pkl'), 'rb'))
    data_to_impute = xgb.DMatrix(df_imputation_ready.to_numpy(dtype = np.float32), nthread = -1, feature_names = df_imputation_ready.columns.to_list())
    imp_res_raw = xgb_trained_model.predict(data_to_impute)
    imp_df = pd.DataFrame(imp_res_raw,columns=le.inverse_transform(range(imp_res_raw.shape[1])))
    imp_df_hc = imp_df.idxmax(axis='columns')
    imp_df.insert(0,'sample_id',list_samples)
    imp_df.insert(1,'chromosome',np.tile(np.array(['chr6_a','chr6_b']),int(imp_df.shape[0]/2)))
    imp_df_hc = pd.DataFrame({'sample_id': list_samples, 'chromosome': np.tile(np.array(['chr6_a','chr6_b']),int(imp_df.shape[0]/2)), 'hard_call': imp_df_hc})
    imp_df.to_csv(os.path.join(args.model_dir, 'imputation_results_'+ args.gene+ '_probabilities.tsv.gz'),sep='\t',index=False)
    imp_df_hc.to_csv(os.path.join(args.model_dir, 'imputation_results_'+ args.gene+ '_hard_calls.tsv.gz'),sep='\t',index=False)
    return 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#this is influenced by deep-hla code too
def main():

    parser = argparse.ArgumentParser(description='Train a model using a HLA reference data.')
    parser.add_argument('--ref_bgl', type=str, required=False, help='HLA reference Beagle format phased haplotype file (.bgl.phased).', dest='ref_bgl')
    parser.add_argument('--ref_bim', type=str, required=False, help='HLA reference extended variant information file (bim format).', dest='ref_bim')
    parser.add_argument('--sample', type=str, required=False, help='Sample SNP data (.bim format).', dest='sample')
    parser.add_argument('--gene',type=str, required=False, help='Gene to train model on (only one), named the same way as in the .model.json file. This is for testing or if reference cohort is very large. Defaults to None', default=None,dest='gene')
    parser.add_argument('--build', type=str, required=False, choices=['grch38','grch37'], default='grch38', help='Chromosomal build, either grch38 (default) or grch37 (currently not supported).', dest='build')
    parser.add_argument('--window',type=int,required=False,help='Extract SNPs around the gene position +/- bp window provided with this option. Only used if --gene is used. Defaults to 500000.',default=500000, dest='window')
    parser.add_argument('--model-dir', type=str,required=False, help='Directory for saving trained models.', dest='model_dir')
    parser.add_argument('--allele_present', type=str, default='T', required=False, choices=['A','C','T','G'], help='Which base pair is chosen to represent presence of the HLA allele (default = T).', dest='allele_present')
    parser.add_argument('--allele_absent',type=str, default='A', required=False, choices=['A','C','T','G'], help='Which base pair is chosen to represent absence of the HLA allele (default = A).', dest='allele_absent')
    parser.add_argument('--use_pandas', type=str2bool, nargs='?', default=False, const=True, choices=[True,False], required=False, help='Whether to use pandas read_table to read the beagle file (the default). If False, then use the open() function to read the file one line at a time and directly encode alleles in a numpy tensor. This requires less memory, but is slower for smaller files.')
    parser.add_argument('--algo_phase',type=str,required=False,help='Which phase of the algorithm: data loading (data_loading), hyperparameter optimization (hyper_opt), xgboost training (xgb_train), predicting (pred).', dest='algo_phase')
    parser.add_argument('--use_gpu',type=str2bool, nargs='?',required=False,const=True, choices=[True,False],help='Whether to use gpus with the cuda engine (True) or not (False, the default). Only used in the cross-validation hyperparameterization optimization step.', default=False,dest='use_gpu')
    parser.add_argument('--threads',type=int,required=False,help='Number of threads to use (Default=1).', default=1,dest='threads')
    parser.add_argument('--nfolds',required=False,type=int,help='Number of folds in 5-fold cross validation (Default=5).', default=5,dest='nfolds')
    parser.add_argument('--cv_seed', required=False,type=int,help='Random seed for cross validation (Default=1).', default=1,dest='cv_seed')
    parser.add_argument('--min_ac',required=False,type=int,help='Minimum HLA allele count to be included in the reference panel (Default=1)', default=1,dest='min_ac')
    parser.add_argument('--snps_for_imputation',required=False,type=str,help='The snps to be used for imputing HLA alleles in a new cohort. Given in bgl.phased format.', dest='snps_for_imputation')
    parser.add_argument('--sample_for_imputation',type=str, required=False, help='SNP data (.bim format) of the SNPs used for imputating the new cohort.', dest='sample_for_imputation')
    parser.add_argument('--two_fields', type=str2bool, nargs='?',required=False,const=True, choices=[True,False], help='Whether to trim at two fields (True) or not (false, the default).',default=False, dest='two_fields')
    parser.add_argument('--weighted_logloss', type=str2bool, nargs='?',required=False,const=True, choices=[True,False], help='Whether to use weighted (True) or unweighted (False, the recommended default) logloss function for training.',default=False, dest='weighted_logloss')
    
    args = parser.parse_args()

    #this block is from deep-hla code
    #BASE_DIR = os.path.dirname(__file__)
    #CUDA_IS_AVAILABLE = torch.cuda.is_available()
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.model_dir+'/logs'):
        os.mkdir(os.path.join(args.model_dir,'logs'))
    
    global logger
    logger = Logger(os.path.join(args.model_dir,'logs/'+ args.algo_phase +'.{}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M'))))
    logger.log('Logging to training.log: '+args.model_dir+'/'+'logs/'+ args.algo_phase+ '.{}.log'.format(datetime.datetime.now().strftime('%y%m%d%H%M')))
    
    
    if args.algo_phase == 'data_loading':
        df_snps,df_hla = load_data(args)
        df_snps_full,df_hla_full = remove_incomplete_haplotypes(df_snps,df_hla,args)
        df_snps_pre_xgb = snps_transpose(df_snps_full)
        df_hla_pre_xgb = collapse_hla(df_hla_full,args)
        df_snps_pre_xgb_mac,df_hla_pre_xgb_mac = min_allele_filter(df_snps_pre_xgb,df_hla_pre_xgb,args)
        df_snps_pre_xgb_binarized = alleles_to_binary(df_snps_pre_xgb_mac)
        prep_for_xgboost(df_snps_pre_xgb_binarized,df_hla_pre_xgb_mac,args)
        return
    
    if args.algo_phase == 'hyper_opt':
        bayes_optim_res = bayes_optim_hla(args)
        return
    
    if args.algo_phase == 'xgb_train':
        xgb_train_hla(args)
        return
    
    if args.algo_phase == 'impute':
        logger.log('XGboost prediction at {}.'.format(time.ctime()))
        xgb_trained_model = xgb_pred_hla(args)
        df_new_snps,list_samples = load_new_snps(xgb_trained_model,args)
        df_new_snps_filled = add_null_columns(df_new_snps,xgb_trained_model)
        df_new_transposed = snps_transpose(df_new_snps_filled,check_variance=False)
        df_imputation_ready = alleles_to_binary(df_new_transposed)
        impute_the_hla(args,df_imputation_ready,xgb_trained_model,list_samples)
        return

if __name__ == '__main__':
    main()
