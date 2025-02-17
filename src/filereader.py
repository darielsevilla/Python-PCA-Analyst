import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        print("\npaso 1")
        print(dataReduced)
    
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

        #print("\nCorrelation Matrix")
        #print(correlationMatrix)
        self.correlationMatrix = correlationMatrix.tolist()
        
        return correlationMatrix
    
  
    def eigen(self):        
        #paso 3: valores y vectores propios
        values, vectors = np.linalg.eig(self.correlationMatrix)
        vectors = vectors.T
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

        #paso 4: union de vectores en matriz
        self.orderedMatrix = vectors.T
        print("\nvectores en matriz")
        print(vectors.T)
        
        self.properValues = values
        self.properVector = vectors.T
        
        
    def principalComponents(self):
        #paso 5: matriz de componentes principales
        rows = self.dataProcessed.values
        print("\nX matrix")
        print(rows[:,1:])
        print()
        pComponents = rows[:,1:]  @ self.orderedMatrix
        print(pComponents)
        self.pComponents = pComponents

#paso 6: matriz de calidades de individuos 
    def individualsQualities(self):
        pComponents = self.pComponents
        rows = self.dataProcessed.values

        #C = pComponents ** 2 #matriz de componentes principales elevado al cuadrado
        iQualities = pComponents ** 2 #matriz de componentes principales elevado al cuadrado
        X = rows[:,1:] ** 2 #matriz de datos procesados al cuadrado

        for i in range(len(X)):
            suma = 0
            for j in range(len(X[i])):
                suma += X[i][j]
            
            for k in range(len(iQualities[i])):
                iQualities[i][k] /= suma

        #iQualities = C / np.sum(X)
        print("iq1\n")
        print(iQualities)
      
        #print(iQualities)

        self.iQualities = iQualities

#paso 7: matriz de coordenada de las variables
    def variablesCoordinates(self):
        diagonal = np.sqrt(np.diag(self.properValues)) #matriz diagonal de la raiz cuadrada de los valores propios
        print("diagonal\n")
        print(diagonal)
        vCoordinates = self.properVector @ diagonal
        print("coordenadas\n")
        print(vCoordinates)
        self.vCoordinates = vCoordinates
        
#paso 8: matriz de calidades de las variables
    def variablesQualities(self):
        vQualities = self.vCoordinates **2 #matriz de coordenadas de las variables elevada al cuadrado
        #print(vQualities)
        self.vQualities = vQualities
       
#paso 9: vector de inercias de los ejes
    def inertiaVector(self):
        #iVector = self.properValues / np.sum(self.properValues)
        iVector = (self.properValues*100)/len(self.properValues)
     
        self.iVector = iVector
        print("inercia\n")
        print(self.iVector)

    def graphCorrelationMatrix(self):
        fig, axis = plt.subplots(figsize=(6, 6)) #tamaño del gráfico
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed', linewidth=1)
        axis.add_patch(circle) #agrega el circulo al gráfico

        #dibuja las lineas de ambos ejes
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        var_coordinates = self.vCoordinates[:, :2]  
        column_names = self.data.columns[1:] #nombre de las materias 

        print("\ncoordenadas")
        print(var_coordinates)

        for i, (x, y) in enumerate(var_coordinates):
            plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='b', ec='b')#dibuja la flecha segun corresponde
            plt.text(x, y, column_names[i], fontsize=12, color='red', ha='center', va='center')#escribe el nombre de la materia
    
        plt.xlim(-1.2, 1.2)#pone limite en los ejes para asegurar que ningun dato quede de fuera
        plt.ylim(-1.2, 1.2)
        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.title("Círculo de Correlación")
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()
