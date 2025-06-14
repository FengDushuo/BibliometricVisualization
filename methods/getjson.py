import os
from csv_to_json import ips_woskeyword

def getall_ips_jsons(inputfiled,outputfiled):
    for root,dirs,files in os.walk(inputfiled):
        for dir in dirs:
            outputfile = outputfiled + "\\" + dir + ".json"
            opendir = inputfiled + "\\" + dir
            for root,dirs,files in os.walk(opendir):
                for file in files:
                    inputfile = root + "\\" + file
                    ips_woskeyword(inputfile,outputfile,"suit")

# getall_ips_jsons("ips_quchong_withlabel","ips_jsons")


