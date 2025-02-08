import pandas as pd
import numpy as np

class FileReader:
    def __init__(self, dir):
        self.dir = dir

    def readData(self):
        data = pd.read_csv(self.dir, sep=";")
        self.data = data

    #paso 1: centrar y reducir
    def centerReduce(self):
        
        rows = self.data.shape[0]
        dataReduced = self.data.copy()
        print(len(self.data.columns))
        print(rows)
        print(dataReduced)
        for i, columnName in enumerate(self.data.columns):
            print("column ", columnName)
           
            changedColumn = []
            if i != 0:
                column = self.data[columnName]
                columnValues = column.tolist()
                #convertir cada elemento de esa columna en numeros
                for i in range(len(columnValues)):
                    columnValues[i] = float(columnValues[i].replace(",","."))
                
                #media de columna
                mean = np.mean(columnValues)
                deviation = np.sqrt(2)

                #centrar y reducir elementos
        
                for item in columnValues:
                    changedColumn.append(float((item-mean)/deviation))
                
                #cambiar por datos actualizados
                #print("changing column ",column, " fin")
                dataReduced[columnName] = changedColumn
                
        self.dataReduced = dataReduced 
        print(dataReduced)       


    

        
