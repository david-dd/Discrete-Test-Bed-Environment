from cmath import inf
import copy
import random as rand
import numpy as np
np.random.seed(0)
from heik import *


class testBedEnvironment:
    
    def __init__(self, uncertainty, ammountOfCarriers):
        self.uncertainty = uncertainty
        self.ammountOfCarriers = ammountOfCarriers
    
    def getOperationTimesWithUncertainty(self, opTimes):
        temp = []
        for x in opTimes:
            temp.append(rand.randint(x-self.uncertainty, x+self.uncertainty))
        return temp

    def setUpEnv(self):
        

        self.log_probs = []
        self.values = []
        self.rewards = []

        self.stepCnt = 0

        self.amountOfOperations = 10

        self.operationsTimes = [
            self.getOperationTimesWithUncertainty([10]),            # Station #0        Operation: 10
            self.getOperationTimesWithUncertainty([17]),            # Station #1        Operation: 20           
            self.getOperationTimesWithUncertainty([35, 35, 35]),    # Station #2        Operation: 20, 30, 40   
            self.getOperationTimesWithUncertainty([16]),            # Station #3        Operation: 30          
            self.getOperationTimesWithUncertainty([15]),            # Station #4        Operation: 40           
            self.getOperationTimesWithUncertainty([22]),            # Station #5        Operation: 50           
            self.getOperationTimesWithUncertainty([22]),            # Station #6        Operation: 50           
            self.getOperationTimesWithUncertainty([20]),            # Station #7        Operation: 60           
            self.getOperationTimesWithUncertainty([35, 12, 35]),    # Station #8        Operation: 60, 70, 80               
            self.getOperationTimesWithUncertainty([13]),            # Station #9        Operation: 80   
            self.getOperationTimesWithUncertainty([16]),            # Station #10       Operation: 90           
            self.getOperationTimesWithUncertainty([18]),            # Station #11       Operation: 90               
            self.getOperationTimesWithUncertainty([10]),            # Station #12       Operation: 100
        ]

        self.stations = [
            [    #-----------------------------------------------------------
                                        # Station #1 (key=0)
                [1],                    # 0 = Operation: 10
                self.operationsTimes[0],      # 1 = Zeit
                1,                      # 2 = PosOnConveyor
                [[]],                   # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #2 (key=1)
                [2],                    # Operation: 20
                self.operationsTimes[1],      # Zeit
                5,                      # PosOnConveyor
                [[2]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #3 (key=2)
                [2,3,4],                # Operation: 20, 30, 40
                self.operationsTimes[2],      # Zeit
                14,                      # PosOnConveyor
                [[1],[3],[4]]           # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #4 (key=3)
                [3],                    # Operation: 30,
                self.operationsTimes[3],      # Zeit
                18,                     # PosOnConveyor
                [[2]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #5 (key=4)
                [4],                    # Operation: 40
                self.operationsTimes[4],      # Zeit
                22,                     # PosOnConveyor
                [[2]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #6 (key=5)
                [5],                    # Operation: 50
                self.operationsTimes[5],      # Zeit
                26,                     # PosOnConveyor
                [[6]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #7 (key=6)
                [5],                    # Operation: 50
                self.operationsTimes[6],      # Zeit
                30,                     # PosOnConveyor
                [[5]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #8 (key=7)
                [6],                    # Operation: 60
                self.operationsTimes[7],      # Zeit
                34,                     # PosOnConveyor
                [[8]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #9 (key=8)
                [6,7,8],                # Operation: 60 ,70, 80
                self.operationsTimes[8],      # Zeit
                38,                     # PosOnConveyor
                [[7],[],[9]],           # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #10 (key=9)
                [8],                    # Operation: 80
                self.operationsTimes[9],      # Zeit
                43,                     # PosOnConveyor
                [[8]],                  # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #11 (key=10)
                [9],                   # Operation: 90
                self.operationsTimes[10],      # Zeit
                47,                     # PosOnConveyor
                [[11]],                 # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #12 (key=11)
                [9],                    # Operation: 90
                self.operationsTimes[11],      # Zeit
                51,                     # PosOnConveyor
                [[10]],                 # 3 = StationNeighbours
            ], [ #-----------------------------------------------------------
                                        # Station #13 (key=12)
                [10],                   # Operation: 100
                self.operationsTimes[12],      # Zeit
                55,                     # PosOnConveyor
                [[]],                   # 3 = StationNeighbours
            ]
        ]

        ################################################################################################
        # Vorher war die Frage: 0=Hier oder 1=beim Nachbarn
        # Das Problem ist: Die Antwort 0="Hier" ist abhänig von der Position
        # Deswegen wird mit dieser tabelle für jede Operation eine feste entscheidungsstruktur definiert 
        self.stationDecisionLookup = []
        for x in range(self.amountOfOperations):
            self.stationDecisionLookup.append([])

        for k, s in enumerate(self.stations):
            for o in s[0]:
                self.stationDecisionLookup[o-1].append(k)
        ################################################################################################

        

        self.carrier = []
        self.carrierHistory = []
        
        # Anzahl an Carriern festlegen
        #amount = rand.randint(self.ammountOfCarriers-self.uncertainty, self.ammountOfCarriers+self.uncertainty)
        amount = self.ammountOfCarriers


        # Entscheidungen für die Stationen initialiseren
        for x in range(amount):
            decisionStations = [
                0,  # Op=10 an Station#1, key=0    
                -1, # Op=20 noch unklar
                -1, # Op=30 noch unklar 
                -1, # Op=40 noch unklar 
                -1, # Op=50 noch unklar 
                -1, # Op=60 noch unklar 
                 8, # Op=70 gibt es nur an Station, key = 8
                -1, # Op=80 noch unklar 
                -1, # Op=90 noch unklar   
                12, # Op=100 an Station#13, key=12        
            ]
            self.carrier.append(
                [           # Carrier #XX
                            1,                  # 0 = nextOp
                            0,                  # 1 = actPos 
                            0,                  # 2 = OpProgress
                            0,                  # 3 = stepCnt-ActionVonStation
                            (x+1),              # 4 = CarrierID
                            0,                  # 5 = CarSollWeiterbewegtWerden
                            0,                  # 6 = stepCnt-ActionVonTransportband
                            decisionStations,   # 7 = individuelle Entscheidungen für jeden Carrier
                ]
            )
            self.carrierHistory.append(
                [           # Carrier #XX
                            [],                 # 0 = HistoryStation
                            [],                 # 1 = Time
                            [],                 # 2 = HistoryOperation
                ]
            )
    


        
        self.aadecisionForAParallelStationNeeded   = []
        self.aadecisionForAParallelStationNeededOp = []
        self.iLastCheckDecisionsNeeded      = -1
        self.popedStationKey                = -1
        self.popedOperation                 = -1


        self.conveyor = [
            # Hier stehen, welche carrierIds sich in den Slots befinden...
            False,  # Pos =  1          Station 1
            False,  # Pos =  2
            False,  # Pos =  3
            False,  # Pos =  4
            False,  # Pos =  5          Station 2
            False,  # Pos =  6
            False,  # Pos =  7          
            False,  # Pos =  8
            False,  # Pos =  9          Station 3
            False,  # Pos =  10
            False,  # Pos =  11
            False,  # Pos =  12
            False,  # Pos =  13         
            False,  # Pos =  14
            False,  # Pos =  15
            False,  # Pos =  16
            False,  # Pos =  17
            False,  # Pos =  18         Station 4
            False,  # Pos =  19        
            False,  # Pos =  20
            False,  # Pos =  21
            False,  # Pos =  22         Station 5
            False,  # Pos =  23
            False,  # Pos =  24
            False,  # Pos =  25
            False,  # Pos =  26         Station 6
            False,  # Pos =  27
            False,  # Pos =  28
            False,  # Pos =  29
            False,  # Pos =  30         Station 7
            False,  # Pos =  31
            False,  # Pos =  32
            False,  # Pos =  33
            False,  # Pos =  34         Station 8
            False,  # Pos =  35
            False,  # Pos =  36
            False,  # Pos =  37
            False,  # Pos =  38         Station 9
            False,  # Pos =  39
            False,  # Pos =  40
            False,  # Pos =  41
            False,  # Pos =  42         Station 10
            False,  # Pos =  43
            False,  # Pos =  44
            False,  # Pos =  45
            False,  # Pos =  46
            False,  # Pos =  47         Station 11
            False,  # Pos =  48
            False,  # Pos =  49
            False,  # Pos =  50
            False,  # Pos =  51         Station 12
            False,  # Pos =  52
            False,  # Pos =  53
            False,  # Pos =  54
            False,  # Pos =  55         Station 13
            False,  # Pos =  56
            False,  # Pos =  57
            False,  # Pos =  58
        ]

        # Carrier zufällig auf das Transportband mappen
        # Zuerst Slots auf dem Transportband bestimmen -> diese erhaten den Wert "-1"
        for i in range(len(self.carrier)):
            foundFreeSlot = False
            while foundFreeSlot != True:
                slotID = rand.randint(1, len(self.conveyor))
                if self.conveyor[slotID-1] == False:
                    # Leeren Slot gefunden, Carrier zuweisen
                    self.conveyor[slotID-1] = -1
                    foundFreeSlot = True

        # Dann alle "-1" austausche durch CarrierID
        # Damit wird gewährleistet, das die Carrier in einer geordneten Reihenfolge sind
        for i in range(len(self.carrier)): 
            oneAdded = False
            for slotKey, slot in reversed(list((enumerate(self.conveyor)))): 
                if self.conveyor[slotKey] == -1 and oneAdded == False:
                    self.conveyor[slotKey] = i+1
                    self.carrier[i][1] = slotKey+1
                    oneAdded = True


        self.conveyorOrg = copy.deepcopy(self.conveyor) 
        self.carrierOrg = copy.deepcopy(self.carrier)
        self.stationsOrg = copy.deepcopy(self.stations)


    def exportStartingConfiguration(self):
        return [
            self.conveyorOrg,
            self.carrierOrg,
            self.stationsOrg
        ]


    def productionFinished(self,carrier):
        retval = True
        for c in carrier:
            if (c[0] != 0):
                #print("Carrier nicht leer", c)
                # nextOp ist nicht null, also müssen noch produkte gefertigt werden
                retval = False
                #print("NOT FINISHED")
                return retval

                


    def getCarrierAtStation(self, keyForStation):
        # Return CarKey
        
        carAtS = False # carKey, oder False  

        # Wir müssen erfahren, für welche Stationen eine Entscheidung getroffen werden muss 
        for k, carIdOnConveyor in (enumerate(self.conveyor)): 
            if carIdOnConveyor != False: 
                # Nur Slot betrachten, in denen sich auch Carrier befinden...
                slotID = k + 1
                carKey = carIdOnConveyor-1

                for stationKey, station in enumerate(self.stations): 
                    if (int(slotID) == int(station[2])):    # conveyorSlot == Position der Station
                        # Der Carrier befindet sich an einer Station
                        if stationKey == keyForStation:
                            # Der Carrier befindet sich an der Station 2
                            carAtS = carKey
        return carAtS

    def checkIfparallelStation(self, stationKey, opperation):
        # ToDo:
        # Hinzufügen: check if the other stations are broken
        # if so, return false -> no alternative existing

        opKey = -1
        for k, op in enumerate(self.stations[stationKey][0]):
            if op == opperation:
                opKey = k
        if opKey == -1:
            # keine paralele Operation
            return False
        else: 
            if len(self.stations[stationKey][3][opKey]) >0:
                return self.stations[stationKey][3][opKey]
            else:
                return False


    def decisionForAParallelStationNeeded(self):

        retVal1 = []
        retVal2 = []

        for k, station in (enumerate(self.stations)): 
            carAtS = self.getCarrierAtStation(k) # get carKey at Station
            if str(carAtS) == "False":
                # Kein Carrier an der Station, also wird hier auch keine Entscheidung benötigt
                pass
            else:
                # Ermittlung der nächsten Operation 
                nextOp = self.carrier[carAtS][0]

                parallelStaions = self.checkIfparallelStation(k, nextOp)

                if parallelStaions == False:
                    # Es gibt keine parallelen Stationen, muss hier also nicht betrachtet werden
                    pass
                else:
                    OpIndex = -1
                    # Ermittlung des OpIndexes der Station
                    for opKey, op in enumerate(station[0]):
                        if op == nextOp: 
                            OpIndex =opKey
                    
                    if OpIndex != -1:
                        # opIndex wurde gefunden.                   
                        # Ermittlung, ob überhaupt eine Nachbarstation existiert  
                        #print("StationKey" , k,  "OpIndex", OpIndex, "nextOp" , nextOp , "station[0]:" , station[0], "carrier" , carAtS )  
                        if len(station[3][OpIndex]) > 0:
                            # Es gibt einen StationNeighbours
                            #print("StationKey" , k,  "Es gibt einen StationNeighbours")    

                            #Ermitteln, ob die Bearbeitung noch nicht begonenn hat        
                            #  NextOp != 0 UND OpProgress == 0
                            if nextOp != 0 and self.carrier[carAtS][2] == 0:
                                # Prüfen, ob die Entscheidungbereits noch nicht getroffen wurde
                                
                                desOpIndex = nextOp-1
                                if self.carrier[carAtS][7][desOpIndex] == -1:

                                    # Ermitteln, ob die nächste op auf dem Carrier auch von der Station angeboten wird
                                    if nextOp in station[0]: # Operation: 30, 40, 50
                                    
                                        # Für diese Station wird nun eine Entscheidung benötigt
                                        #print("Entscheiung benötigt für StationKey" , k , " mit nextOp" , nextOp )
                                        retVal1.append(k)           # Stationkeys
                                        retVal2.append(nextOp)      # nextOp

        return [retVal1,retVal2]
        


    def getFollowingOperation(self,actOp):
        retVal = 0
        if actOp == 0:
            retVal = 0
        else:
            retVal = actOp + 1
            if retVal == 11:
                retVal = 0
        return retVal
    
    def shouldTheNextOperationExecuted(self, nextOp, stationKey, carrierKey):
        #print(self.executeOpForCarrier)
        if nextOp == 0:
            return False
        else:           
            # Prüfen, ob die Staion die Operation überhaupt ausführen kann
            #        Operation An Station   == die nächste Operation die auf dem Carrier ausgeführt werden soll
            if nextOp in self.stations[stationKey][0]:
                # Die Operation kann an der Station ausgeführt werden.

                # Wenn es keine Alternativen gibt, oder die alternatiiven broken sind, dann immer ausführen
                StationNeighbours = self.checkIfparallelStation(stationKey, nextOp)  
                if StationNeighbours == False:
                    # immer ausführen
                    return True
                else:
                    # Es gibt Nachbarstationen, also nachschalgen, welche Entscheidung für diese getroffen wurde:
                    # Nachschlagen in der Entscheidungsliste (hängt am Carrier) 
                    
                    # Der Inhalt ist
                    # -1  = Noch keine Entscheidung getroffen
                    # key = oder der key der Station, die die Operation ausführen soll
                    if self.carrier[carrierKey][7][nextOp-1] == stationKey:          
                        # In der Entscheidungsliste steht die Station drin, an der sich der Carrier gerade befindet, also ausführen
                        return True
                    else:
                        # Der StationKey stimmt nicht mit der ZielStation überein, also hier nicht fertigen!
                        return False
            else:
                # Die Operation kann hier nicht ausgeführt werden... 
                return False
    
    def getIndividualWaitingstimes(self):
        iMin = inf
        iMax = -inf
        allWainting = []
        avgWait = 0

        for ch in self.carrierHistory:
            temp = 0
            for hist in ch[0]:
                if hist == "W":
                    temp += 1
            allWainting.append(temp)


        for w in allWainting:
            if w < iMin:
                iMin = w
            if w > iMax:
                iMax = w
        
        avgWait = np.mean(allWainting)
        return [iMin, avgWait, iMax, allWainting]

    def calcReward(self,duration):

        ############################################################
        ############################################################
        ### Reward ohne Aufwendige Berechnung
        ############################################################
        ############################################################

        overallWaiting = 0
        ############################################################
        # 1. Alle Wartezeiten bestimmen
        ############################################################
        for ch in self.carrierHistory:
            for hist in ch[0]:
                if hist == "W":
                    overallWaiting += 1



        lastWaiting = 0
        ############################################################
        # 2. Wartezeiten des letzten Carriers bestimmen
        ############################################################
        for hist in self.carrierHistory[-1][0]:
            if hist == "W":
                lastWaiting += 1

        ############################################################   
        # 3. Parallelezeiten ermittln
        ############################################################


        
        # zunächst ermitteln, welche paralele Operationen existieren
        operations=[2,3,4,5,6,7,8]
        """
        for s in self.stations:
            if len(s[3]) >0:
                # Min eine Nachbarstation vorhanden
                if (s[0] in operations) == False:
                    operations.append(s[0]) 
        # print("paralelle Operationen",  operations)
        """
        operationTimes = {}

        # Für jede Paralelel Station den Zähler auf 0 setzten
        for o in operations:
            operationTimes.update({o: 0})

        # Nun für jeden Zeitschritt gucken, ob Operationen paralele gelaufen sind
        for i in range(duration):
            temp=[]
            for ch in self.carrierHistory:
                temp.append(ch[2][i])

            for o in operations:
                c = temp.count(o)
                if c >= 2:
                    operationTimes.update({o: (operationTimes[o]+1)})

        # Alle operationszeiten zusammen zählen
        overallparalelTimes = 0
        x = operationTimes.values()
        for pt in x:
            overallparalelTimes += pt

        # tested RewardFunctions (for small Env)
        # V2
        # reward2 = (100 - overallWaiting) + (100-(3*lastWaiting)) + (overallparalelTimes*10) 
        # V3
        #reward2 = ((100 - overallWaiting)*2) + (overallparalelTimes*5) 
        
        # V4
        reward = ((0-overallWaiting)*5) + (overallparalelTimes*3) 


        #print("overallWaiting=" , overallWaiting)
        #print("lastWaiting=" , lastWaiting)
        #print("overallparalelTimes=" , overallparalelTimes)
        #print("reward2", reward2)

        return [reward, overallWaiting, overallparalelTimes]

    def getActualState(self):

        # 00000 conveyor slot is empty
        # 10000 carrier without a nextOp on slot
        # 10001 carrier with nextOp=1 in slot
        # 10010 carrier with nextOp=2 in slot
        # 10011 carrier with nextOp=3 in slot
        # 10100 carrier with nextOp=4 in slot
        # 10101 carrier with nextOp=5 in slot
        # 10110 carrier with nextOp=6 in slot
        # 10111 carrier with nextOp=7 in slot
        # 11000 carrier with nextOp=8 in slot
        # 11001 carrier with nextOp=9 in slot
        # 11010 carrier with nextOp=10 in slot

        retval = []
        for k, conv in enumerate(self.conveyor):
            if conv == False:
                # conveyor slot is empty
                retval.append(0)  # Car in Slot
                retval.append(0)  # nextOp -> 1. digit  2
                retval.append(0)  # nextOp -> 2. digit  4
                retval.append(0)  # nextOp -> 3. digit  8
                retval.append(0)  # nextOp -> 4. digit  16
            else:
                # der aktuelle Slot ist nicht leer -> also steht hier eine CarID drin
                # Dass wird durch das Erste Bit angezeigt
                retval.append(1)  # Car in Slot

                carID = conv
                # Nun muss in dem Array "carrier" nachgeschlagen werden, was die nächste OP ist
                nextOp = int(self.carrier[carID-1][0])  # 0 = nextOp
                strOp = intToBinary(nextOp, 4)    # Op die an der Station angeboten werden
                for x in strOp:
                    retval.append(int(x)) 

        retval = np.array(retval) #convert to np Array
        return retval
    
    def startAnEpisode(self):
        self.setUpEnv()
        return self.stepUntilNextDecision() #Finished, Reward, actualState

    def startATrainEpisode(self, conveyor=False, carrier=False, stations=False):
        self.setUpEnv()

        if conveyor != False or carrier != False or stations != False:
            # Nur Überschreiben, wenn die Daten auch übergeben wurden
            self.conveyor = conveyor
            self.carrier = carrier
            self.stations = stations
        
        return self.stepUntilNextDecision() #Finished, Reward, actualState

    def startAnEvalEpisode(self, conveyor, carrier, stations):
        self.setUpEnv()
        self.conveyor = conveyor
        self.carrier = carrier
        self.stations = stations
        
        return self.stepUntilNextDecision() #Finished, Reward, actualState

    def step(self, action):
        # Entscheidung von der KI
        try:
            action = bool(action.item()) 
        except:
            pass

        anfragendeStation =  self.popedStationKey  
        anfragendeOperation = self.popedOperation

        carAtS = self.getCarrierAtStation(anfragendeStation) 

        #Nun muss bei der zugehörigen Station nachgeschlagen werden was diese Entscheidung bedeutet
        # Früher galt:
        # 0 = ausführen bei der Anfragenden Station
        # 1 = ausführend bei der Nachbarstation
        # Jetzt zählt der index
        # False = Index 0
        # True = Index 1


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


        ps = self.stationDecisionLookup[anfragendeOperation-1]
        
        if action == False:
            stationKey = ps[0]  # linke Spalte
        else:
            stationKey = ps[1]  # rechte Splate
        self.carrier[carAtS][7][anfragendeOperation-1] = stationKey
                
        #print("die Entscheiudngen für dne Carrier mit Key=", carAtS, "Entscheidungen", str(self.carrier[carAtS][7]))
        return self.stepUntilNextDecision() #Finished, Reward, actualState
        
    def getOperationIndex(self, stationKey, operation):
        opKey = -1
        for k, op in enumerate(self.stations[stationKey][0]):
            if op == operation:
                opKey = k
        return opKey


    def getOperationTime(self, stationKey, operation):
        opkey = self.getOperationIndex(stationKey, operation)
        if opkey == -1:
            die("fehlerhafter Opkey -getOperationTime")
            return False
        else:
            return self.stations[stationKey][1][opkey]
        
    def stepUntilNextDecision(self):
        #Gibt folgendes zurück
        # 0 = Finished
        # 1 = Reward
        # 2 = actualState

        while self.productionFinished(self.carrier) == False:  
            #print(self.stepCnt)   

            # Abfrage, ob neue Entscheidungen benötigt werden
            if self.stepCnt >= self.iLastCheckDecisionsNeeded:
                self.aadecisionForAParallelStationNeeded, self.aadecisionForAParallelStationNeededOp = self.decisionForAParallelStationNeeded()
                self.iLastCheckDecisionsNeeded = self.stepCnt+1
                #print("self.aadecisionForAParallelStationNeeded", self.aadecisionForAParallelStationNeeded)
                temp = []
                for c in self.carrier:
                    temp.append(c[0])
                #print(self.stepCnt, temp)

            if len(self.aadecisionForAParallelStationNeeded) > 0:
                self.popedStationKey = self.aadecisionForAParallelStationNeeded.pop()
                self.popedOperation = self.aadecisionForAParallelStationNeededOp.pop()

                # eine Entscheidung wird benötigt, also aktuellen Zustand erfassen und Antwort abholen.
                envState = self.getActualState()
                encodeedStationKey = intToOneHotEncodedString(self.popedStationKey, len(self.stations)) 
                #print("Asking Station key=", self.popedStationKey, "encodeedStationKey=", encodeedStationKey)
                returnState = envState
                #print("self.popedStationKey", self.popedStationKey)
                #print("returnState", returnState)
                for x in encodeedStationKey:
                    if str(x) == str(0):
                        returnState = np.append(returnState, 0)
                    else:
                        returnState = np.append(returnState, 1)  
  
                #print("ENTSCHEIDUNG wird benötigt")
                #print("encodeedStationKey" ,self.popedStationKey,  encodeedStationKey)
                #print(envState)

                """
                carAtS = self.getCarrierAtStation(self.popedStationKey)
                print("Entscheidung für Carrier:", carAtS, "an Station", self.popedStationKey )     
                tempStr = str(carAtS) + "-" + str(self.popedStationKey)
                if tempStr in self.debugArrayAnfragenAntworten:
                    print(self.stepCnt, "Antwort wiederverwendet!!!" , tempStr,"=", self.debugArrayAnfragenAntworten[tempStr], self.carrier[carAtS])
                    self.step(self.debugArrayAnfragenAntworten[tempStr])
                """
                return [False, 0, returnState, [self.popedStationKey, self.popedOperation]]
            else:
                 
                
                self.stepCnt += 1
                # Update der Stationen
                for k, carIdOnConveyor in (enumerate(self.conveyor)): 
                    if carIdOnConveyor != False:             
                        # Nur Slot betrachten, in denen sich auch Carrier befinden...    
                        carKey = carIdOnConveyor-1       
                        lastCarUpdate = self.carrier[carKey][3]


                        if lastCarUpdate < self.stepCnt:
                            slotID = k + 1
                            nextSlotID = slotID +1
                            # In diesem Schritt wurde noch keine Aktionfür den Carrier ausgeführt
                            

                            self.carrier[carKey][5] = False         # CarSollWeiterbewegtWerden
                            if nextSlotID > len(self.conveyor):
                                nextSlotID = 1
                            

                            # befindet sich der Carrier an einer Station?
                            carAtStation = False 
                            for stationKey, station in enumerate(self.stations): 
                                if (int(slotID) == int(station[2])):
                                    carAtStation = True
                                    break # Wir haben eine Station gefunden, an weiteren Stationen kann der Carrier nicht sein

                            
                            if carAtStation == True:
                                # Der Carrer befindet sich an einer Station
                                nextOp = self.carrier[carKey][0]
                                # wird gerade eine Operation auf dne Carrier angewendet?
                                if self.carrier[carKey][2] > 0:
                                    # Auf dem Carrier wird eine Operation ausgeführt!
                                    # Ist die Operation vorbei?
                                    if self.carrier[carKey][2] >= self.getOperationTime(stationKey, nextOp):
                                    #if self.carrier[carKey][2] >= station[1]:
                                        # Operation ist vollendet, der Carrier kann die Station verlassen...
                    

                                        opFinished = True                        
                                        nextOp = self.getFollowingOperation(nextOp)
                                        self.carrier[carKey][0] = nextOp            # Set the following operation
                                        self.carrier[carKey][2] = 0                 # Reset Progress
                                        self.carrier[carKey][3] = self.stepCnt      # Carrier wurde behandelt...
                                        self.carrier[carKey][5] = True              # CarSollWeiterbewegtWerden
                                        
                                    else:
                                        # Operation noch nicht vollendet, also Progress erhöhen...
                                        self.carrier[carKey][2] = self.carrier[carKey][2]+1               
                                        #print("Carrier:",car, "Ausführen der Operation" , nextOp, "Fortschritt" , carrier[carKey][2])             
                                        self.carrier[carKey][3] = self.stepCnt    # Carrier wurde behandelt...

                                        self.carrierHistory[carKey][0].append(str(stationKey))
                                        self.carrierHistory[carKey][1].append(self.stepCnt)
                                        self.carrierHistory[carKey][2].append(nextOp)
                                else:
                                    # Auf dem Carrier wird noch keine Operation angeboten
                                    # Kann und soll die Station die nächste operation ausführen?

                                    executed = self.shouldTheNextOperationExecuted(nextOp, stationKey, carKey)                                  
                                    if executed == True:
                                        # Operation soll ausgeführt werden, also starten..
                                        #print("Carrier:",car, "Ausführen der Operation" , nextOp, "Fortschritt" , 1)
                                        self.carrier[carKey][2] = 1
                                        self.carrier[carKey][3] = self.stepCnt    # Carrier wurde behandelt...
                                        #print("t=", self.stepCnt, "EXECUTED",  "nextOp", nextOp, "stationKey", stationKey,  "carKey", carKey)

                                        self.carrierHistory[carKey][0].append(str(stationKey))
                                        self.carrierHistory[carKey][1].append(self.stepCnt)
                                        self.carrierHistory[carKey][2].append(nextOp)
                                    else:
                                        # operation, soll nicht ausgeführt werden, also weiterschicken... 
                                        self.carrier[carKey][3] = self.stepCnt      # Carrier wurde behandelt...
                                        self.carrier[carKey][5] = True              # CarSollWeiterbewegtWerden

                            else:
                                # Der Carrier befindt sich an keiner Station, also einfach weitertransporieren...
                                self.carrier[carKey][3] = self.stepCnt      # Carrier wurde behandelt...
                                self.carrier[carKey][5] = True              # CarSollWeiterbewegtWerden

                # Update des Transportbandes
                for a in range(2):
                    for k, carIdOnConveyor in (enumerate(self.conveyor)): 
                        if carIdOnConveyor != False: 
                            carKey = carIdOnConveyor-1
                            lastConveyorUpdate = self.carrier[carKey][6]    

                            if lastConveyorUpdate < self.stepCnt:

                                nextConveyorKey = k + 1
                                if nextConveyorKey > len(self.conveyor)-1:
                                    nextConveyorKey = 0


                                # den Carrier weitertransporieren, wenn gewünscht und Slot vor ihm nicht belegt ist ist.
                                if self.carrier[carKey][5] == True and self.conveyor[nextConveyorKey] == False:
                                    self.conveyor[k]                 = False                # Aktuellen Slot leeren
                                    self.conveyor[nextConveyorKey]   = carIdOnConveyor      # nächsten Slot mit dem Wert aus dem aktuellen Slot setzten
                                    self.carrier[carKey][6] = self.stepCnt                  # Carrier wurde behandelt...
                                    if (self.stepCnt in self.carrierHistory[carKey][1]) == False:
                                        self.carrierHistory[carKey][0].append("T")              # History (transport)
                                        self.carrierHistory[carKey][1].append(self.stepCnt)
                                        self.carrierHistory[carKey][2].append("T")
                                elif self.carrier[carKey][5] == True and self.conveyor[nextConveyorKey] != False:
                                    #self.carrier[carKey][6] = self.stepCnt                  # Carrier wurde behandelt... # gefährlich her?
                                    if (self.stepCnt in self.carrierHistory[carKey][1]) == False:
                                        if self.carrier[carKey][0] == False:
                                            # nextOp = 0 -> Der Carrier ist fertig..           
                                            self.carrierHistory[carKey][0].append("F") 
                                            self.carrierHistory[carKey][1].append(self.stepCnt)
                                            self.carrierHistory[carKey][2].append("F")
                                        else:
                                            self.carrierHistory[carKey][0].append("W") 
                                            self.carrierHistory[carKey][1].append(self.stepCnt)
                                            self.carrierHistory[carKey][2].append("W")

        returnState = self.getActualState() 
        for x in range(len(self.stations)):
            returnState = np.append(returnState, 0)           
        return [True, self.stepCnt, returnState, [False, False]] 
