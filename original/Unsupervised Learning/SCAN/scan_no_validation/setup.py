import os 
import pandas as pd 
import numpy as np 
import argparse 


args = argparse.ArgumentParser(description="Setup File")
args.add_argument("--data_dir",help = "Path to the data")
args = args.parse_args()



def get_data_dict(data_dir):
 
        data_dict = {'path': [], 'label': []}
        
        for  label in os.listdir(data_dir):
           
                label_dir = data_dir + label + '/'
                for files in os.listdir(label_dir):
                    
                        data_dict['path'].append(label_dir + files)
                        data_dict['label'].append(label)
 
        df =  pd.DataFrame(data_dict)
        #shuffling the dataframe.
        df = df.sample(frac = 1).reset_index(drop = True)
        n = len(df)
        #set 75 for training and   and 25 for testing 
        
 
        df['train'] = np.asarray([np.random.choice([1,0],1,p = [0.75,0.25]) for _ in range(n)])
 

        return df



if __name__ == "__main__":
    
    data_path = args.data_dir
    print(f'loading data from {data_path}')
    df = get_data_dict(data_path)

    df.to_csv("data/data_df.csv",index = False)
    print('created .csv file in /data dir')