from tkinter import S
import gym
import numpy as np
from random import *               

import os
import numpy as np
import matplotlib.pyplot as plt
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

from torch.distributions import Categorical

import ast      # Convert String from MySQL to Array

###########################################################
###########################################################
### Import custom Libraries
###########################################################
###########################################################

import sys
sys.path.append(os.path.join('C:/', 'Users','NAME','Desktop','YOUR','PATH','Discrete-Test-Bed-Environment','_libs'))
from envTestBed import *
from heik import *  

def start(c, u, dec): 

    ###########################################################
    ###########################################################
    ### Settings
    ###########################################################
    ###########################################################

    ###########################################################
    ### Training Setting
    ###########################################################

    state_space = (5*58) + 13    # ((Operationscodierung+CarrierCodierung)*Anzahl Plätze auf Transportband) + Anzahl Stationen = 303
    action_space = 2

    minEpisodes                     = 200 
    maxEpisodes                     = 20000
    avgOverLastRuns                 = 25
    checkpointAfter                 = 50



    ###########################################################
    # PPO-LSTM
    gamma       = 0.99                    
    gae_lambda  = 0.95      # Smoothing parameter
    policy_clip = 0.2       # Sorgt dafür, dass der Loss angehoben wird


    N           = 24        # N = 2048 -> horizon, number of steps before we perform an update
    batch_size  = 12
    n_epochs     = 4
    learning_rate= 0.0003    # learning_rate, alpha

    ###########################################################
    ### Env Setting
    ###########################################################


    uncertainty                     = u
    ammountOfCarriers               = c

    
    ###########################################################
    ### Eval Setting
    ###########################################################
    
    ammountOfDatasetsEval   = 1000
    envVersion              = 1
    
    ###########################################################
    ### Setup Env and Agent
    ###########################################################

    env = testBedEnvironment(uncertainty,ammountOfCarriers)

    ###########################################################
    ### Init
    ###########################################################
 
    OpRoundRobin=[
        False,      # Op10
        False,      # Op20
        False,      # Op30
        False,      # Op40
        False,      # Op50
        False,      # Op60
        False,      # Op70
        False,      # Op80
        False,      # Op90
        False,      # Op100
         ]

    ###########################################################
    ###########################################################
    ### Start Eval
    ###########################################################
    ###########################################################

    # Idee - erst eine kleine Evaluierung für 10-50 Datasets, dann eine große über 100-10000 Datasets
    evalMakespan = []
    evalReward = []
    datasets = getDatasets(ammountOfCarriers, uncertainty, envVersion)
    evalCnt = 0
    for d in datasets:
        #print("evalCnt" , evalCnt)
        #  0        , 1       , 2        
        # `conveyor`,`carrier`,`stations`
        try:
            conveyor    = ast.literal_eval(d[0])
            carrier     = ast.literal_eval(d[1])
            stations    = ast.literal_eval(d[2])
        except:
            conveyor    = ast.literal_eval(d[0].decode("utf-8"))
            carrier     = ast.literal_eval(d[1].decode("utf-8"))
            stations    = ast.literal_eval(d[2].decode("utf-8"))
            
        done, duration, state, [stationKey, opKey] = env.startAnEvalEpisode(conveyor,carrier,stations)
        

        j = 0
        while not done:
            # Was bedeutet False/True?
            # -> Beschreibt den Indes von der stationDecisionLookup (enthält die StationKeys)
            #           
            #       Entscheidung: 
            #           False, True
            #Op10		[0      ]
            #Op20 		[1,    2]
            #Op30		[2,    3]
            #Op40		[2,    4]
            #Op50		[5,    6]
            #Op60		[7,    8]
            #Op70		[8      ]
            #Op80		[8,    9]
            #Op90		[10,   11]
            #Op100		[12    ]



            if dec == 0:
                action = True #random.choice([True, False])
            if dec == 1:
                # 0 Shortest Path
                action = False #random.choice([True, False])   
            if dec == 2:
                # Random
                action = random.choice([True, False])       
            if dec == 3:
                # Fastes
                #print(stationKey, opKey)
                action = False
                if opKey == 0:
                    action = False  # Op10
                elif opKey == 1:
                    action = False  # Op20
                elif opKey == 2:
                    action = True   # Op30
                elif opKey == 3:
                    action = True   # Op40
                elif opKey == 4:
                    action = False  # Op50
                elif opKey == 5:
                    action = False  # Op60
                elif opKey == 6:
                    action = False  # Op70
                elif opKey == 7:
                    action = True   # Op80
                elif opKey == 8:
                    action = False  # Op90
                elif opKey == 9:
                    action = False  # Op100
            
            if dec == 4:
                # Fastes
                #print(stationKey, opKey)
                action = False
                if opKey == 0:
                    action = False  # Op10
                elif opKey == 1:
                    action = True  # Op20
                elif opKey == 2:
                    action = False   # Op30
                elif opKey == 3:
                    action = False   # Op40
                elif opKey == 4:
                    action = False  # Op50
                elif opKey == 5:
                    action = True  # Op60
                elif opKey == 6:
                    action = False  # Op70
                elif opKey == 7:
                    action = False  # Op80
                elif opKey == 8:
                    action = True  # Op90
                elif opKey == 9:
                    action = False  # Op100
            if dec == 5:
                if opKey == 0:
                    action = False  # Op10
                elif opKey == 6:
                    action = False  # Op70
                elif opKey == 9:
                    action = False  # Op100
                else:
                    action = OpRoundRobin[opKey]
                    if action == True:                        
                        OpRoundRobin[opKey] = False
                    else:
                        OpRoundRobin[opKey] = True

       
            done, duration, nextState, [stationKey, opKey] = env.step(action)   
            state = nextState
            j = j +1

        reward = env.calcReward(duration)
        evalReward.append(reward)
        evalMakespan.append(duration)
    
    avg_eval_reward = np.mean(evalReward[-ammountOfDatasetsEval:])
    avg_eval_makespan = np.mean(evalMakespan[-ammountOfDatasetsEval:])

    print("Evaluierung des Modells mit",len(datasets), "/",ammountOfDatasetsEval, "Datensätzen", "avgReward:" ,  avg_eval_reward, "avgMakespan:" , avg_eval_makespan)



###########################################################
###########################################################
### Start Training Process
###########################################################
###########################################################

if __name__ == '__main__':

    aAmmountOfCarriers = [
        13
    ]

    aUncertainty = [
        3 
    ]


    j = 0


    for x in [2]:
        print("##########################################")
        print("##########################################")
        print("##########################################")
        print("Zustand=" ,x , "0=True, 1=False (shortest), 2=random, 3=fastest, 4=Slowest, 5=Roound Robin")
        print("##########################################")
        print("##########################################")
        print("##########################################")
        for c in aAmmountOfCarriers:
            for u in aUncertainty:
                for i in range(1):
                    res = False
                    print("#####################################")
                    print(j, "ammountOfCarriers", c, "uncertainty=", u)
                    print("#####################################")
                    res = start(c, u, x)

                    
                    j = j+1
