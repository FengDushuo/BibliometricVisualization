import csv
import json

def ips_woskeyword(openfile,outputfile,markertype):
    resultdict = {"links":[]}
    with open(openfile,"r") as csvfile:
        lines = csvfile.readlines()
        for line in lines:
            line.replace("\n", "").replace('\r', '')
            linelist = line.split(",")
            linedict = {"source":linelist[0],"target":linelist[1].replace("\n", ""),"type":markertype}
            resultdict["links"].append(linedict)
    json_str = json.dumps(resultdict)

    with open(outputfile,"w") as json_file:
        json_file.write(json_str)
