# -- coding:utf-8 --
import os
import pandas as pd
import numpy as np
import pdb
import argparse

def mergeFile(file_list1,file_list2, out_addr ):
#     file1 = open("2.lst", "r",encoding='UTF-8')
#     file2 = open("1.lst", "r",encoding='UTF-8')
#     file_list1 = file1.readlines() 
#     file_list2 = file2.readlines() 
    file_list1=file_list1['Addr']
    file_list2=file_list2['Addr']
    file_list=[]
    for i in range(file_list1.__len__()):
        a=str(file_list1[i])
        b=str(file_list2[i])
        file_list.append(a + ' ' + b)
    df = pd.DataFrame(file_list, columns=['one'])
    df.to_csv(out_addr, columns=['one'], index=False, header=False)
    # file = open("train_pair.lst", "w")
    # file.writelines(file_list)
#     file1.close()
#     file2.close()
    # file.close()


def ReadSaveAddr(InputStra, InputStrb, out_addr, test=False):
    for dirpath,dirnames,filenames in os.walk(InputStra):
        file_list=pd.Series(filenames).apply(lambda x: dirpath+x)
        file_list=file_list.apply(lambda x: x.split('/', 7)[-1])
        filenames_len=filenames.__len__()
        if filenames_len:
            df1 = pd.DataFrame(np.arange(filenames_len).reshape((filenames_len,1)),columns=['Addr'])
            df1.Addr = file_list
    if test == True:
        df1.to_csv(out_addr,columns=['Addr'],index=False,header=False)
    else:
        for dirpath,dirnames,filenames in os.walk(InputStrb):
            file_list=pd.Series(filenames).apply(lambda x: dirpath+x)
            file_list=file_list.apply(lambda x: x.split('/', 7)[-1])
            filenames_len=filenames.__len__()
            if filenames_len:
                df2 = pd.DataFrame(np.arange(filenames_len).reshape((filenames_len,1)),columns=['Addr'])
                df2.Addr = file_list
        mergeFile(df1,df2, out_addr)
    print("Write To Get.lst !")


def main():
    # Get arguments
    parser = argparse.ArgumentParser(description='welcome to the web crawling programme for splitting image to 1024*1024')
    parser.add_argument('-path', '--path', type=str,
        help='path of input data dir  ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel')
    parser.add_argument('-path_out', '--path_out', type=str,
        help='data out abs address for picture ex:C:\\Users\\heca0002\\Desktop\\GIS\\map_example\\data_excel ')   

    args = parser.parse_args()
    test=True
    ReadSaveAddr(args.path, '', args.path_out, test=test)

if __name__ == '__main__':
    main()
    '''
    InputStra="/home/GDDC-CV1/Desktop/data_1024_hrnet/data/urbanisation/x/test/"
    InputStrb=""
    out_addr='/home/GDDC-CV1/Desktop/data_1024_hrnet/data/list/urbanisation/test.lst'
    test=True
    ReadSaveAddr(InputStra, InputStrb, out_addr, test=test)
    '''

'''
Input Example:
InputStra="/home/GDDC-CV1/Desktop/data_1024_hrnet/data/urbanisation/x/train/"
InputStrb="/home/GDDC-CV1/Desktop/data_1024_hrnet/data/urbanisation/y/train/"
out_addr='/home/GDDC-CV1/Desktop/data_1024_hrnet/data/list/urbanisation/train.lst'
test=True - used in testing set
'''
