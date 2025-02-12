import filereader as File

def main():
    fileReader = File.FileReader("dataset\EjemploEstudiantes.csv")
    fileReader.readData()
    #fileReader.createRelationsMatrix()
    fileReader.createRelationsMatrix2()
    fileReader.eigen()
    fileReader.principalComponents()

if __name__ == '__main__':
    main()
    