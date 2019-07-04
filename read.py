import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import pandas as pd

# read the dataset and return the all customer IDs
def save_IDs(path,savepath):
  raw = pd.read_csv(path)
  log = raw['message']
  customer_id=[]
  for row in log:
      st = row.index('for customer: ')
      temp = row[st+14:st+20]
      if not temp in customer_id:
          customer_id.append(temp)
  np.save(savepath,customer_id)

# read dataset 
train_path = './train.csv'
train_id_save_path = './customer_id.npy'
test_path = './test.csv'
test_id_save_path = './test_customer_id.npy'
save_IDs(train_path,train_id_save_path)
save_IDs(test_path,test_id_save_path)

customer_id = np.load(train_id_save_path)

## tranform format of dataset, save it in table format in features below
FEATURES = ['quote_start_time','quote_finish_time','quote_unfinish_time','pay_time','gender','name','age','address','email','square_footage','number_of_floors',
             'number_of_bedrooms','type','way','household_num','claim','amount']
for i in range(10):
  FEATURES.append('house_age_'+str(i))
  FEATURES.append('house_gender_'+str(i))
  FEATURES.append('house_name_'+str(i))
df = pd.DataFrame(columns=FEATURES, index = customer_id,data = 0)

data = pd.read_csv(train_path)
for i in range(len(data)): 
  cur_data = data['message'][i].split('-')
  infor = cur_data[2]
  for j in range(3,len(cur_data)):
    infor = infor+cur_data[j]
  id_index = infor.find('for customer:')
  if id_index<0:
    import pdb
    pdb.set_trace()
  cus_cur_id = infor[id_index+14:id_index+20]
  
  index = infor.find('Quote Started')
  time = data['timestamp'][i]
  if index>0:
    df.loc[cus_cur_id,'quote_start_time'] = time
  index = infor.find('Quote Completed')
  if index>0:
    df.loc[cus_cur_id,'quote_finish_time'] = time
  index = infor.find('Quote Incomplete')
  if index>0:
    df.loc[cus_cur_id,'quote_unfinish_time'] = time
  index = infor.find('Payment Completed')
  if index>0:
    df.loc[cus_cur_id,'pay_time'] = time

  index = infor.find('with json payload')
  if index>0:
    string = infor[index+17:].replace("'",'"')
    feature_raw = json.loads(string)
    try:
      df.loc[cus_cur_id,'gender'] = feature_raw['gender']
    except:
      pass
    try:
      df.loc[cus_cur_id,'name'] = feature_raw['name']
    except:
      pass
    try:
      df.loc[cus_cur_id,'age'] = feature_raw['age']
    except:
      pass
    try:
      df.loc[cus_cur_id,'address'] = feature_raw['address']
    except:
      pass
    try:
      df.loc[cus_cur_id,'email'] = feature_raw['email']
    except:
      pass
    try:
      df.loc[cus_cur_id,'square_footage'] = feature_raw['home']['square_footage']
    except:
      pass
    try:
      df.loc[cus_cur_id,'number_of_floors'] = feature_raw['home']['number_of_floors']
    except:
      pass
    try:
      df.loc[cus_cur_id,'number_of_bedrooms'] = feature_raw['home']['number_of_bedrooms']
    except:
      pass
    try:
      df.loc[cus_cur_id,'type'] = feature_raw['home']['type']
    except:
      pass
    try:
      house_num = len(feature_raw['household'])
    except:
      continue
    if house_num>10:
      import pdb
      pdb.set_trace()
    df.loc[cus_cur_id,'household_num'] = house_num
    for house_hold_index in range(house_num):
      try:
        df.loc[cus_cur_id,'house_age_'+str(house_hold_index)] = feature_raw['household'][house_hold_index]['age']
      except:
        pass
      try:
        df.loc[cus_cur_id,'house_gender_'+str(house_hold_index)] = feature_raw['household'][house_hold_index]['gender']
      except:
        pass
      try:
        df.loc[cus_cur_id,'house_name_'+str(house_hold_index)] = feature_raw['household'][house_hold_index]['name']
      except:
        pass

  index = infor.find('Claim Accepted')
  if index>0:
    index = infor.find('paid')
    amount = infor[index+6:]
    df.loc[cus_cur_id,'amount'] = amount
  index = infor.find('Claim Started for customer:')
  if index>0:
    df.loc[cus_cur_id,'claim'] = 1
  infor = cur_data[1]
  df.loc[cus_cur_id,'way'] = infor
  if i%1000==0:
    print(i)
print('Tranform finished!')
df.to_csv('./train_cleaned.csv')

# see the output
print(data.head())