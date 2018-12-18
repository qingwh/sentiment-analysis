#!/usr/bin/python

"""
To use, please type in:

    python concatenate.py

The script will only look at files that are within
folders that are one level below the directory supplied
to the script. For example, if no argument is given to
the script, script will parse all folders within the 
current working directory as such:

    ./output_folder1/output_file1
    ./output_folder1/output_file2
    ./output_folder2/output_file1
    ./output_folder2/output_file2
    ./output_folder_n/output_file_n

"""

import pandas as pd
import sys
import re
import os

def main():
    # combine all files in all subdirs
    combineFiles(getUserInputFile())

    # add headers, indicies, remove tuple parentheses
    df = pd.read_csv('./combinedFile.csv', delimiter=',', quotechar='"')
    headerLabels = ['url','iphone','samsunggalaxy','sonyxperia','nokialumina','htcphone','ios','googleandroid','iphonecampos','samsungcampos','sonycampos','nokiacampos','htccampos','iphonecamneg','samsungcamneg','sonycamneg','nokiacamneg','htccamneg','iphonecamunc','samsungcamunc','sonycamunc','nokiacamunc','htccamunc','iphonedispos','samsungdispos','sonydispos','nokiadispos','htcdispos','iphonedisneg','samsungdisneg','sonydisneg','nokiadisneg','htcdisneg','iphonedisunc','samsungdisunc','sonydisunc','nokiadisunc','htcdisunc','iphoneperpos','samsungperpos','sonyperpos','nokiaperpos','htcperpos','iphoneperneg','samsungperneg','sonyperneg','nokiaperneg','htcperneg','iphoneperunc','samsungperunc','sonyperunc','nokiaperunc','htcperunc','iosperpos','googleperpos','iosperneg','googleperneg','iosperunc','googleperunc']
    df.columns = headerLabels
    df.index.name = 'id'

    # output factor and url files
    df.to_csv('concatenated_websites.csv', columns=headerLabels[:1], quotechar='"', sep=',',header=True)
    df.to_csv('concatenated_factors.csv', columns=headerLabels[1:], quotechar='"', sep=',',header=True)

    # cleanup
    os.remove('combinedFile.csv')
    print ("Sucessfully processed " + str(fileCount) + " files")
    sys.exit()

def getUserInputFile():
    file = input("Enter input file path (blank for current directory): ")
    if file == "":
        file = "."
    return file

def combineFiles(file):
    outfile = open('combinedFile.csv', 'w+')
    global fileCount
    fileCount = 0
    httpRe = re.compile(r'.*?[http]')
    for dirname, dirnames, filenames in os.walk(file):
        # For each sub folder
        for subdirname in dirnames:
            subdirpath = os.path.join(dirname, subdirname)
            for fileName in os.listdir(subdirpath):
                fileCount += 1
                print ("Processing " + fileName + "...")
                with open(subdirpath + "/" + fileName) as infile:
                    for line in infile:
                        # make sure we're reading reducer output files
                        if len(httpRe.findall(line)) > 0:
                            outfile.write(line)
        return None

if __name__ == "__main__":
    main()

