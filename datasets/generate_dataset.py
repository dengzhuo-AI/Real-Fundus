import os, random, shutil
def generate_dataset(fileDir,tarDir,rate):
    pathDir = os.listdir(fileDir+"gt/")
    filenumber=len(pathDir)
    # select_rate
    rate=rate
    picknumber=int(filenumber*rate)
    sample = random.sample(pathDir, picknumber)
    print (sample)
    for name in sample:
        if not os.path.exists(tarDir+"gt/"):
            os.makedirs(tarDir+"gt/")
        if not os.path.exists(tarDir+"input/"):
            os.makedirs(tarDir+"input/")
        shutil.move(fileDir+"gt/"+name, tarDir+"gt/"+name)
        shutil.move(fileDir + "input/" + name, tarDir + "input/" + name)
    return

if __name__ == '__main__':
    rfdir = "./Real_Fundus/"
    testdir = "./test/"
    valdir = "./val/"
    traindir = "./train/"
    generate_dataset(rfdir,testdir,0.25)
    generate_dataset(rfdir, valdir, 0.1)
    generate_dataset(rfdir, traindir, 1)
