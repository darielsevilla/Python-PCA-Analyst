import filereader as File

def main():
    fileReader = File.FileReader("dataset\EjemploEstudiantes.csv")
    fileReader.readData()
    #fileReader.createRelationsMatrix()
    fileReader.createRelationsMatrix2()
    fileReader.eigen()
    print("Principal Components")
    fileReader.principalComponents()
    print("\nIndividual Qualities")
    fileReader.individualsQualities()
    print("\nVariables Coordinates")
    fileReader.variablesCoordinates()
    print("\nVariables Qualities")
    fileReader.variablesQualities()
    print("\nInertia Vector")
    fileReader.inertiaVector()
    fileReader.graphCorrelationMatrix(0,1)
    #fileReader.graphCorrelationMatrix(3,4)

if __name__ == '__main__':
    main()
    