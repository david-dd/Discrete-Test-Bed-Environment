import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import mysql.connector              # MYSQL
import random


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)
    plt.close('all')

def die(text = ""):
    print("############################################################################")
    print()
    print()
    print()
    raise ValueError(text)

def getCurrentTime():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M_%S")
    return str(current_time + "")

def intToBinary(input, lenght):
    output = str(bin(int(input))).replace("0b", "")

    while len(output)<lenght:
        output = "0" + output 

    if output.endswith("-1"):
        output = output[:len(output) - 2]
        output = "1" + str(output) + "0"

    return output

def intToOneHotEncodedString(input, lenght):
    input = int(input)

    output = ""
    while len(output)<lenght:
        output = "0" + output 

    output = replace_str_index(output, input, "1")

    return output[::-1] # reverse

def replace_str_index(text,index=0,replacement=''):
    return '%s%s%s'%(text[:index],replacement,text[index+1:])


###########################################################
### MySQL Setting
###########################################################

mysqlHost = "localhost"
mysqlUser = "root"
mysqlPassword = ""
mysqlDatabase = "testbed"

#####################################################################
#####################################################################
# MySQL Functions
#####################################################################
#####################################################################

def openMySQL(mysqlHost, mysqlUser, mysqlPassword, mysqlDatabase):

    mydb = mysql.connector.connect(
        host=mysqlHost,
        user=mysqlUser,
        password=mysqlPassword,
        database=mysqlDatabase
    )
    mycursor = mydb.cursor(prepared=True)
    return [mydb, mycursor]


def closemySQL(mydb):
    mydb.close()


#####################################################################
#####################################################################
# Eval-Funktionen
#####################################################################
#####################################################################

def createRandomSortedList(num, start = 1, end = 100): 
    arr = [] 
    tmp = random.randint(start, end) 
      
    for x in range(num):           
        while tmp in arr: 
            tmp = random.randint(start, end)               
        arr.append(tmp)           
    arr.sort() 
      
    return arr 
   

def getDatasets(ammountOfCarriers, uncertainty, version = 1):
    global mysqlHost, mysqlUser, mysqlPassword, mysqlDatabase

    #Open MySql
    mydb, mycursor = openMySQL(mysqlHost, mysqlUser, mysqlPassword, mysqlDatabase)


    sql = "SELECT `conveyor`,`carrier`,`stations` FROM `evalsetlarge` WHERE `version` = " + str(version) + " and `ammountOfCarriers` = " + str(ammountOfCarriers) + " and `uncertainty` = " + str(uncertainty) +";"
    mycursor.execute(sql)
    retVal = mycursor.fetchall()

    # DB wieder schließen        
    closemySQL(mydb)

    return retVal
