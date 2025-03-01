import filereader as File

def main():
    fileReader = File.FileReader("dataset\EjemploEstudiantes.csv")
    fileReader.readData()
    fileReader.createRelationsMatrix2()
    fileReader.eigen()
    fileReader.principalComponents()
    fileReader.individualsQualities()
    fileReader.variablesCoordinates()
    fileReader.variablesQualities()
    fileReader.inertiaVector()
    #fileReader.graphCorrelationMatrix(0,1)
    #fileReader.graphCorrelationMatrix(3,4)
    fileReader.graphPrincipalPlane(0,1)
    fileReader.graphPrincipalPlane(3,4)

if __name__ == '__main__':
    main()
    