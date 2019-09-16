#!/usr/bin/env python

'''
Name : Omkar Thawakar

Following code build the decision tree with ID3 Algorithm.

Code require confog file with folowing format

{
   'data_file' : './data/data.csv',
   'data_mappers' : [],
   'data_project_columns' : [
    'label','cap-shape','cap-surface','cap-color','bruises','gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'
    ],
   'target_attribute' : 'label'
}


'''

import ast
import csv
import sys
import math
import os
import pandas as pd


def load_csv_to_header_data(filename):

    fs = csv.reader(open(filename, newline='\n'))

    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }
    return data

def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def project_columns(data, columns_to_project):
    data_h = list(data['header'])
    data_r = list(data['rows'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h, 'rows': data_r,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name}


def get_uniq_values(data):
    idx_to_name = data['idx_to_name']
    idxs = idx_to_name.keys()

    val_map = {}
    for idx in iter(idxs):
        val_map[idx_to_name[idx]] = set()

    for data_row in data['rows']:
        for idx in idx_to_name.keys():
            att_name = idx_to_name[idx]
            val = data_row[idx]
            if val not in val_map.keys():
                val_map[att_name].add(val)
    return val_map


def get_class_labels(data, target_attribute):
    rows = data['rows']
    col_idx = data['name_to_idx'][target_attribute]
    labels = {}
    for r in rows:
        val = r[col_idx]
        if val in labels:
            labels[val] = labels[val] + 1
        else:
            labels[val] = 1
    return labels


def entropy(n, labels):
    ent = 0
    for label in labels.keys():
        p_x = labels[label] / n
        ent += - p_x * math.log(p_x, 2)
    return ent


def partition_data(data, group_att):
    partitions = {}
    data_rows = data['rows']
    partition_att_idx = data['name_to_idx'][group_att]
    for row in data_rows:
        row_val = row[partition_att_idx]
        if row_val not in partitions.keys():
            partitions[row_val] = {
                'name_to_idx': data['name_to_idx'],
                'idx_to_name': data['idx_to_name'],
                'rows': list()
            }
        partitions[row_val]['rows'].append(row)
    return partitions


def avg_entropy_w_partitions(data, splitting_att, target_attribute):
    # find uniq values of splitting att
    data_rows = data['rows']
    n = len(data_rows)
    partitions = partition_data(data, splitting_att)

    avg_ent = 0

    for partition_key in partitions.keys():
        partitioned_data = partitions[partition_key]
        partition_n = len(partitioned_data['rows'])
        partition_labels = get_class_labels(partitioned_data, target_attribute)
        partition_entropy = entropy(partition_n, partition_labels)
        avg_ent += partition_n / n * partition_entropy

    return avg_ent, partitions


def most_common_label(labels):
    mcl = max(labels, key=lambda k: labels[k])
    return mcl


def id3(data, uniqs, remaining_atts, target_attribute,depth=None):
    labels = get_class_labels(data, target_attribute)

    node = {}

    if len(labels.keys()) == 1:
        node['label'] = next(iter(labels.keys()))
        return node

    if len(remaining_atts) == 0:
        node['label'] = most_common_label(labels)
        return node

    n = len(data['rows'])
    ent = entropy(n, labels)

    max_info_gain = None
    max_info_gain_att = None
    max_info_gain_partitions = None

    for remaining_att in remaining_atts:
        avg_ent, partitions = avg_entropy_w_partitions(data, remaining_att, target_attribute)
        info_gain = ent - avg_ent
        if max_info_gain is None or info_gain > max_info_gain:
            max_info_gain = info_gain
            max_info_gain_att = remaining_att
            max_info_gain_partitions = partitions

    if max_info_gain is None:
        node['label'] = most_common_label(labels)
        return node

    node['attribute'] = max_info_gain_att
    node['Info-Gain'] = max_info_gain
    node['nodes'] = {}
    
    ##############################################
    
    ##############################################
    
    remaining_atts_for_subtrees = set(remaining_atts)
    remaining_atts_for_subtrees.discard(max_info_gain_att)

    uniq_att_values = uniqs[max_info_gain_att]

    for att_value in uniq_att_values:
        if att_value not in max_info_gain_partitions.keys():
            node['nodes'][att_value] = {'label': most_common_label(labels)}
            continue
        partition = max_info_gain_partitions[att_value]
        if depth == None:
            node['nodes'][att_value] = id3(partition, uniqs, remaining_atts_for_subtrees, target_attribute)
        elif depth == 0 :
            return node
        else:
            node['nodes'][att_value] = id3(partition, uniqs, remaining_atts_for_subtrees, target_attribute,depth-1)

    return node



def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    return ast.literal_eval(data)

#################################################################################
def get_label(root,example):
    if 'label'in root.keys():
        return (root['label'])
    else:
        try:
            return get_label(root['nodes'][example[root['attribute']]],example)
        except:
            return None
    
def get_depth(root,depth):
    if 'nodes' in root.keys():
        depth+=1
        return get_depth(root['nodes'],depth+1)
    else:
        print(depth)
################################################################################

def pretty_print_tree(root):
    stack = []
    rules = set()

    def traverse(node, stack, rules):
        if 'label' in node:
            stack.append(' THEN ' + node['label'])
            rules.add(''.join(stack))
            stack.pop()
        elif 'attribute' in node:
            ifnd = 'IF ' if not stack else ' AND '
            stack.append(ifnd + node['attribute'] + ' EQUALS ')
            for subnode_key in node['nodes']:
                stack.append(subnode_key)
                traverse(node['nodes'][subnode_key], stack, rules)
                stack.pop()
            stack.pop()

    traverse(root, stack, rules)
    print(os.linesep.join(rules))


config = load_config(sys.argv[1])


data = load_csv_to_header_data(config['data_file'])
data = project_columns(data, config['data_project_columns'])

target_attribute = config['target_attribute']
remaining_attributes = set(data['header'])
remaining_attributes.remove(target_attribute)

uniqs = get_uniq_values(data)

root = id3(data, uniqs, remaining_attributes, target_attribute, depth=5)

print('='*50)
print('Constructed Decision Tree is given below >>>>>>> ')
print(root)
print('='*50)
print('Rules derived from constructed tree are :: ')
pretty_print_tree(root)
print('='*50)



# # 3. Limiting Depth

# a.Run 5-fold cross-validation using the specified files. Experiment with
# depths in the set 1, 2, 3, 4, 5, 10, 15, reporting the average cross-validation
# accuracy and standard deviation for each depth. Explicitly specify which depth
# should be chosen as the best, and explain why.

# In[4]:


def cross_validation_accuracy(file):
    file = pd.read_csv(file)
    correct,total=0,len(file)
    for i in range(len(file)):
        example = {
                    'cap-shape':file['cap-shape'][i],
                    'cap-surface':file['cap-surface'][i],
                    'cap-color':file['cap-color'][i],
                    'bruises':file['bruises'][i],
                    'gill-attachment':file['gill-attachment'][i],
                    'gill-spacing':file['gill-spacing'][i],
                    'gill-size':file['gill-size'][i],
                    'gill-color':file['gill-color'][i],
                    'stalk-shape':file['stalk-shape'][i],
                    'stalk-root':file['stalk-root'][i],
                    'stalk-surface-above-ring':file['stalk-surface-above-ring'][i],
                    'stalk-surface-below-ring':file['stalk-surface-below-ring'][i],
                    'stalk-color-above-ring':file['stalk-color-above-ring'][i],
                    'stalk-color-below-ring':file['stalk-color-below-ring'][i],
                    'veil-type':file['veil-type'][i],
                    'veil-color':file['veil-color'][i],
                    'ring-number':file['ring-number'][i],
                    'ring-type':file['ring-type'][i],
                    'spore-print-color':file['spore-print-color'][i],
                    'population':file['population'][i],
                    'habitat':file['habitat'][i]
                  }

        if get_label(root,example) == file['label'][i] :
            correct+=1

    return (correct/total)*100


cross_folds = ['cross_fold_1234','cross_fold_1235','cross_fold_1245','cross_fold_1345','cross_fold_2345']
depths = [1, 2, 3, 4, 5, 10, 15]
for i in range(len(cross_folds)):
    acc = []
    for depth in depths:
        config = load_config('data/CVfolds/fold{}.cfg'.format(len(cross_folds)-i))
        data = load_csv_to_header_data(config['data_file'])
        data = project_columns(data, config['data_project_columns'])

        target_attribute = config['target_attribute']
        remaining_attributes = set(data['header'])
        remaining_attributes.remove(target_attribute)

        uniqs = get_uniq_values(data)

        root = id3(data, uniqs, remaining_attributes, target_attribute,depth=depth)
        
        print('='*50)
        acc.append(cross_validation_accuracy('data/CVfolds/fold{}.csv'.format(len(cross_folds)-i)))
        print('Depth : ',depth)
        print('Cross Validation Accuracy on fold{}.csv ::: {} '              .format(len(cross_folds)-i , cross_validation_accuracy('data/CVfolds/fold{}.csv'.format(len(cross_folds)-i))))
    print('#'*50)  
    print('Max Accuracy is {} for fold {} .'.format(max(acc),depths[acc.index(max(acc))]))
        
    print('#'*50)





