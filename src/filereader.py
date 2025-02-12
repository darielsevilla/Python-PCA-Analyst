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
        
        for i, columnName in enumerate(self.data.columns):
            
           
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
                  
                    changedColumn.append(float((item-mean)/deviation))
           
                #cambiar por datos actualizados
                dataReduced[columnName] = changedColumn
        self.dataProcessed = dataReduced    
    
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
    
    def createRelationsMatrix2(self):
        #para pasos 1
        self.centerReduce()
        
        #empezando el paso 2
        #consiguiendo matriz transpuesta
        rows = self.dataProcessed.values
        
        columns = self.dataProcessed.columns

        transposed = rows[:,1:].T 
        correlationMatrix = transposed @ rows[:,1:]


        #numero escalar que hay que multiplicar
        scalar = 1/(len(rows))
        correlationMatrix = correlationMatrix*scalar

        self.correlationMatrix = correlationMatrix.tolist()
        
        return correlationMatrix
    
  
    def eigen(self):        
        #paso 3: valores y vectores propios
        values, vectors = np.linalg.eig(self.correlationMatrix)
        
        #ordenamiento junto con los vectores
        for i in range(len(values)):
            for j in range(i, len(values)):
                if(values[j] >  values[i]):
                    vector_val1 = np.copy(vectors[i])
                    vector_val2 = np.copy(vectors[j])
                    
                    vectors[i] = vector_val2
                    vectors[j] = vector_val1
                    
                    val = values[i]
                    values[i] = values[j]
                    values[j] = val 

        #print(values)
        print(vectors)

        #paso 4: union de vectores en matriz
        self.orderedMatrix = vectors.T
        
        

    def principalComponents(self):
        #paso 5: matriz de componentes principales
        
        
        rows = self.dataProcessed.values
        print()
        pComponents = rows[:,1:]  @ self.orderedMatrix
        print(pComponents)