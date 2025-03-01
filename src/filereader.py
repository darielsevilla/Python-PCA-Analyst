import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class FileReader:
    def __init__(self, dir):
        self.dir = dir

    def readData(self):
        data = pd.read_csv(self.dir, sep=";")
        self.data = data
      
    def printMatrix(self, matrix):
        for i in range(len(matrix)):
            print("[", end=" ")
            for j in range(len(matrix[0])):
                print(str(matrix[i][j]), end=" ")
            print("]")    

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
        print("\nPaso #1 - Matriz Centrada y Reducida")
        print(dataReduced)
    

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

        print("\nPaso #2 - Matriz de Correlaciones")
        self.printMatrix(correlationMatrix)
        self.correlationMatrix = correlationMatrix.tolist()
        
        return correlationMatrix

    def eigen(self):        
        #paso 3: valores y vectores propios
        values, vectors = np.linalg.eig(self.correlationMatrix)
        sorted_indices = np.argsort(values)[::-1]
    
        print("\nPaso #3 - Vectores y Valores Propios")
        print("Vectores Propios")
        print(vectors)
        print("\nValores propios")
        print(values)

        #ordenamiento junto con los vectores
        for i in range(len(values)):
            for j in range(i, len(values)):
                if(values[j] >  values[i]):
                    # Intercambiar valores propios
                    values[i], values[j] = values[j], values[i]
            
                    # Intercambiar vectores propios
                    temp = np.copy(vectors[:, i])
                    vectors[:, i] = vectors[:, j]
                    vectors[:, j] = temp

        #paso 4: union de vectores en matriz
        self.properValues = values
        self.properVector = vectors
        
        print("\nPaso #4 - Matriz de Vectores Propios")
        print(self.properVector)
        
    def principalComponents(self):
        #paso 5: matriz de componentes principales
        rows = self.dataProcessed.values

        pComponents = rows[:,1:]@self.properVector
        print("\nPaso #5 - Matriz de Componentes Principales")
        self.printMatrix(pComponents)
        self.pComponents = pComponents

    #paso 6: matriz de calidades de individuos 
    def individualsQualities(self):
        pComponents = self.pComponents
        rows = self.dataProcessed.values
        
        iQualities = pComponents ** 2 #matriz de componentes principales elevado al cuadrado
        X = rows[:,1:] ** 2 #matriz de datos procesados al cuadrado
        
        for i in range(len(iQualities)):
            suma = 0
            for j in range(len(X[i])):
                suma += X[i][j]
            
            for k in range(len(X[i])):
                iQualities[i][k] /= suma
    
        print("\nPaso #6 - Matriz de Calidades de Individuos")      
        self.printMatrix(iQualities)

        self.iQualities = iQualities

    #paso 7: matriz de coordenada de las variables
    def variablesCoordinates(self):
        diagonal = np.sqrt(np.diag(self.properValues)) #matriz diagonal de la raiz cuadrada de los valores propios
        
        vCoordinates = np.dot(diagonal,self.properVector)

        print("\nPaso #7 - Matriz de Coordenadas de la Variables")
        print(vCoordinates)
        self.vCoordinates = vCoordinates

    #paso 8: matriz de calidades de las variables
    def variablesQualities(self):
        vQualities = (self.vCoordinates**2)
        fil, col = vQualities.shape
        for i in range(fil):
            for j in range(col):
                vQualities[i][j] /= self.properValues[j]

        print("\nPaso #8 - Matriz de Calidades de las Variables")
        self.printMatrix(vQualities)
        self.vQualities = vQualities
       
    #paso 9: vector de inercias de los ejes
    def inertiaVector(self):
        iVector = (self.properValues*100)/len(self.properValues)
        self.iVector = iVector

        print("\nPaso #9 - Vector de Inercias de los Ejes")
        print(self.iVector, end="\n\n")

    def graphCorrelationMatrix(self, dim1, dim2):
        fig, axis = plt.subplots(figsize=(6,6)) #tamaño del gráfico
        circle = plt.Circle((0, 0), 1, color='teal', fill=False, linestyle='dashed', linewidth=1)
        axis.add_patch(circle) #agrega el circulo al gráfico

        #dibuja las lineas de ambos ejes
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        coordinates = self.vCoordinates[:, [dim1, dim2]]
        norms = np.linalg.norm(coordinates, axis=1, keepdims=True)
        coordinates = coordinates / norms
        courses = self.data.columns[1:] #nombre de las materias 
        
        for i, (x, y) in enumerate(coordinates):
            plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='teal', ec='teal')#dibuja la flecha segun corresponde
            plt.text(x, y, courses[i], fontsize=12, color='brown', ha='center', va='center')#escribe el nombre de la materia
    
        plt.xlim(-1.2, 1.2)#pone limite en los ejes para asegurar que ningun dato quede de fuera
        plt.ylim(-1.2, 1.2)

        plt.xlabel(f"Componente {dim1}", fontsize=12, color='black')
        plt.ylabel(f"Componente {dim2}", fontsize=12, color='black')
        plt.title(f"Círculo de Correlación (Dim {dim1}, {dim2})")
        
        for i, (x, y) in enumerate(coordinates):
            plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color='teal', width=0.005)
            #plt.quiver(0, 0, x, y, head_width=0.05, head_length=0.05, fc='teal', ec='teal')#dibuja la flecha segun corresponde
            plt.text(x, y, courses[i], fontsize=12, color='brown', ha='center', va='center')#escribe el nombre de la materia
    
        plt.xlim(-1.2, 1.2)#pone limite en los ejes para asegurar que ningun dato quede de fuera
        plt.ylim(-1.2, 1.2)

        plt.xlabel(f"Componente {dim1}", fontsize=12, color='black')
        plt.ylabel(f"Componente {dim2}", fontsize=12, color='black')
        plt.title(f"Círculo de Correlación (Dim {dim1}, {dim2})")

        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.show()

    def graphPrincipalPlane(self, dim1, dim2):
        fig, axis = plt.subplots(figsize=(10, 6))  # Tamaño del gráfico
        max_val = np.max(np.abs(self.vCoordinates[:, [dim1, dim2]]))

        if dim1 == 0 and dim2 == 1:
            plt.xlim(-2.5, 3.5)
            plt.ylim(-2.0, 2.0)
        else:
            plt.xlim(-0.70, 0.70)
            plt.ylim(-0.20, 0.20)

        # Extraer los dos primeros componentes principales
        pComponents = self.pComponents[:, [dim1,dim2]]  
        nombres = self.data.iloc[:, 0]  # Nombres de los individuos

        # Graficar cada individuo en el plano principal
        plt.scatter(pComponents[:, 0], pComponents[:, 1], color='purple', alpha=0.7, label="Personas")

        # Agrega los nombres a cada punto
        for i, (x, y) in enumerate(pComponents):
            plt.text(x, y, nombres[i], fontsize=9, color='black', ha='right', va='bottom')

        plt.axhline(0, color='gray', linewidth=0.7, linestyle="dashed")
        plt.axvline(0, color='gray', linewidth=0.7, linestyle="dashed")

        plt.xlabel(f"Componente {dim1}", fontsize=12, color='black')
        plt.ylabel(f"Componente {dim2}", fontsize=12, color='black')
        plt.title(f"Plano Principal (Dim {dim1}, {dim2})", fontsize=14, color='black')

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.show()