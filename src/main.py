import filereader as File

def main():
    fileReader = File.FileReader("dataset\EjemploEstudiantes.csv")
    fileReader.readData()
    fileReader.createRelationsMatrix()

if __name__ == '__main__':
    main()
    