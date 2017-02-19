import cv2

from scipy import ndimage

from vektor import distance
from vektor import pnt2line2
from skimage.measure import label
from skimage.measure import regionprops
from sklearn.datasets import fetch_mldata

from skimage import color
import numpy as np

#Danilo Dimitrijevic RA71/2013
#genericki projekat za ocene 7 i 8



def pronadjiLiniju(videoName):
    cap = cv2.VideoCapture(videoName)
    checkVideo = True

    while checkVideo:
        checkVideo = False
        minx = 0
        miny = 0
        maxy = 0
        maxx = 0
        while(cap.isOpened()):
            
            ret, frame = cap.read()
            print("Ucitao video")
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #kernel = np.ones((2,2),np.uint8)
                #cv2.dilate(gray, kernel)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
       
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, 600, 8)
        if(videoName == "video/video-0.avi" or videoName =="video/video-3.avi"):
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 600, 8)
#        
        for x1, y1, x2, y2 in lines[0]:
            minx=x1
            miny=y1
            maxx=x2
            maxy=y2
    
        #for i in  range(len(lines)):

        i = 0
        while i < len(lines):
            for x1, y1, x2, y2 in lines[i]:
                if x1<minx :
                    print "minimum " + format(i)
                    minx=x1
                    miny=y1
                if x2>maxx:
                    print "maximum " + format(i)
                    maxy=y2
                    maxx=x2
                i = i + 1
                cv2.line(frame, (minx,miny), (maxx, maxy), (0, 255, 0), 2)
        #cv2.line(frame, (minx,miny), (maxx, maxy), (0, 255, 0), 2)
        cv2.imshow('sli', frame)

        return minx,miny, maxx,maxy
    
def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['centar'], obj['centar'])
        if(mdist<r):
            retVal.append(obj)
    return retVal
    
    
    
def pronadjiNajblizeg(list,elem):

    temp = list[0]
    for obj in list:
        if distance(obj['centar'],elem['centar']) < distance(temp['centar'],elem['centar']):
            temp = obj
    
    return temp
    
def leviUgao(img_gray,string,minx,miny,maxx,maxy):
    try:
        img = label(img_gray)
        regions = regionprops(img)
        newImg = np.zeros((28, 28))
        for region in regions:
            bbox = region.bbox
            if bbox[0]<minx:
                minx=bbox[0]
            if bbox[1] <miny:
                miny=bbox[1]
            if bbox[2]>maxx:
                maxx=bbox[2]
            if bbox[3]>maxy:
                maxy=bbox[3]

    
        height = maxx - minx
        width = maxy - miny
    
        newImg[0:height, 0:width] = newImg[0:height, 0:width] + img_gray[minx:maxx, miny:maxy]
                    
        return newImg
    except ValueError:
        print "prodjiiii"
        pass
    
def pronadjiBroj(img_BW,minx,miny,maxx,maxy):
    try:
        label_img = label(img_BW)
        regions = regionprops(label_img)
        if len(regions) > 1:
            print "vece i "+ format(len(regions))
        newImg = np.zeros((28, 28))
        for region in regions:
            bbox = region.bbox
            if bbox[0]<minx:
                minx=bbox[0]
            if bbox[1] <miny:
                miny=bbox[1]
            if bbox[2]>maxx:
                maxx=bbox[2]
            if bbox[3]>maxy:
                maxy=bbox[3]

    
        height = maxx - minx
        width = maxy - miny
    
        newImg[0:height, 0:width] = newImg[0:height, 0:width] + img_BW[minx:maxx, miny:maxy]
        
        return newImg
    except ValueError:
        print "sssssssssss"
        pass

def uporediMnist(newImg):
#    cv2.imshow("nova slika", newImg)
    i=0;
    minSum = 9999
    rez = -1
    for i in range(len(mnist.data)):
       sum=0
       mnist_img=lista_mnist[i]
       sum=np.sum(mnist_img!=newImg)
       if sum < minSum:
           minSum = sum
           rez = mnist.target[i]
       
    #print "broj ----  " + format(rez)
    return rez
     
def preuzmiBroj(img):

    img_BW=color.rgb2gray(img) >= 0.88
    img_BW=(img_BW).astype('uint8')
    newImg=pronadjiBroj(img_BW, 500,500,-1,-1)
    
    rez = uporediMnist(newImg)
    
    return rez;


def napuniMnist(mnist):

    i=0;
    print "mnist data " + format(len(mnist.data))
    for i in range(len(mnist.data)):
        mnist_img=mnist.data[i].reshape(28,28)
        mnist_temp = color.rgb2gray(mnist_img)/255.0 >= 0.88
        mnist_img_gray=(mnist_temp).astype('uint8')
        new_mnist_img=leviUgao(mnist_img_gray,"sss",500,500,-1,-1)
        lista_mnist.append(new_mnist_img)
        

lista_mnist=[]
mnist = fetch_mldata('MNIST original')
cc = -1
minx=700
miny=700
maxx=-1
maxy=-1

videoName="video/video-0.avi"
cap = cv2.VideoCapture(videoName)

x1,y1,x2,y2=pronadjiLiniju(videoName)
linija = [(x1, y1), (x2, y2)]

predjenih = 0
sumaPredjenih = []
def main():
    
    

    boundaries = [
        ([220, 220, 220], [255, 255, 255])
    ]

    elements = []
    t =0
    napuniMnist(mnist)
    while (1):
        
        (lower, upper) = boundaries[0]

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        
        ret, img = cap.read()
        if not ret:
            break
        
        
        mask = cv2.inRange(img, lower, upper)
        img0 = 1.0 * mask
        
        img01 = 1.0 * mask
       
        kernel = np.ones((2,2),np.uint8)
        img0 = cv2.dilate(img0, kernel)
        img0 = cv2.dilate(img0, kernel)
        
        
        #img0 = update2(img0)
        #pronalazi objekte koji su pronadjeni na slici i jedinstveno ih oznacava i sadrzani su
        #u objektu labeled, a broj pronadjenih objekata se nalazi u promjenljivij nr_objects
        labeled, nr_objects = ndimage.label(img0)
        objects = ndimage.find_objects(labeled) 
        for i in range(nr_objects):
            loc = objects[i]
            (xc, yc) = ((loc[1].stop + loc[1].start) / 2,
                        (loc[0].stop + loc[0].start) / 2)
            (dxc, dyc) = ((loc[1].stop - loc[1].start),
                          (loc[0].stop - loc[0].start))

            if (dxc > 11 or dyc > 11):

                elem = {'centar': (xc, yc), 'size': (dxc, dyc), 't': t}
                
                lst = inRange(20, elem, elements)
                nn = len(lst)
                if nn == 0:
                    x1=xc-14
                    x2=xc+14
                    y1=yc-14
                    y2=yc+14
                    global cc
                    cc = cc + 1
                    elem['id'] = cc
                    elem['vrednost'] = preuzmiBroj(img01[y1:y2,x1:x2])
                    elem['t'] = t
                    elem['prosao'] = False
                    
                    elements.append(elem)
                else:
                    el = pronadjiNajblizeg(lst,elem)
                    el['centar'] = elem['centar']
                    el['t'] = t
                

        for el in elements:
            tt = t - el['t']
            if (tt < 3):
                dist, pnt, r = pnt2line2(el['centar'], linija[0], linija[1])
                if r > 0:

                    if (dist < 7):
                        if el['prosao'] == False:
                            el['prosao'] = True
                            
                            (x,y)=el['centar']

                            x1=x-14
                            x2=x+14
                            y1=y-14
                            y2=y+14

                            print "presao broj:  " + format(el['vrednost'])
                           # cv2.waitKey()
                            global predjenih
                            global sumaPredjenih
                            sumaPredjenih.append(el['vrednost'])
                            predjenih = predjenih + 1


        t += 1

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        suma = sum(sumaPredjenih)
    print "zbir " + format(suma)
    print "predjenih " + format(predjenih)

    cap.release()
    cv2.destroyAllWindows()

main()