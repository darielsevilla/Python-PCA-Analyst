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
                numerator = 0
                #desviacion estandar
                for j in range(len(columnValues)):
                    numerator += ((columnValues[j]-mean)*(columnValues[j]-mean))

                deviation = np.sqrt(numerator/len(columnValues))
                
                #centrar y reducir elementos
                for item in columnValues:
                    print(item, " ")
                    changedColumn.append(float((item-mean)/deviation))
                print("\n")
                #cambiar por datos actualizados
                dataReduced[columnName] = changedColumn
                
                
        self.dataProcessed = dataReduced 
        #print(dataReduced)       
    
    def printMatrix(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                print("[",matrix[i][j],"]",end="")
            print("")

    def createRelationsMatrix(self):
        #para pasos 1
        self.centerReduce()

        #empezando el paso 2
        #consiguiendo matriz transpuesta
        rows = self.dataProcessed.values
        print(rows)
        columns = self.dataProcessed.columns

        transposed = np.empty((len(columns), len(rows)))  
        
        for i, row in enumerate(rows):
            for j in range(len(columns)):
                if(isinstance(row[j],float)):
                    transposed[j][i] = row[j]
                else:
                    transposed[j][i] = None
           

        #numero escalar que hay que multiplicar
        
        scalar = 1/(len(rows))
        print(self.dataProcessed)

        
        #consiguiendo la matriz de correlaciones
        correlationMatrix = []
        for i in range(1,len(transposed)):
            row = []
            for j in range(1,len(columns)):
                #multiplicacion matricial
                item = 0
                for k in range(len(transposed[0])):
                    item += (transposed[i][k] * rows[k][j]*scalar)   
                row.append(item)
            correlationMatrix.append(row)
        
        self.correlationMatrix = correlationMatrix                
        return correlationMatrix
    


    
        
