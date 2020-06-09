import cv2 as cv;
import os;
import numpy as np;

winSize=(30, 50);
blockSize=(10, 10);
blockStride=(5, 5);
cellSize=(5, 5);
nbins=9;

total=0;
correct=0;

anno_path_dir="./data/test/annotations";
image_path_dir="./data/test/images";

anno_file_list=os.listdir(anno_path_dir);
image_file_list=os.listdir(image_path_dir);

print("loading svm model");
svm=cv.ml.SVM_load("digits_svm_model.yml");
print("finished loading svm model");

print("hog started");
hog=cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
print("finished hog");

print("started test");
for anno in anno_file_list:
    anno_fd=open(os.path.join(anno_path_dir, anno), 'r');
    pos=[];
        
    anno_line=anno_fd.readline();
    while anno_line:
        anno_line=anno_fd.readline();
        #print(anno_line);
        if not anno_line: break;

        pos.append(anno_line.strip().split(' '));

    img=cv.imread(os.path.join(image_path_dir, anno.replace("txt", "jpg")),
           cv.IMREAD_GRAYSCALE);

    for p in pos:
        total+=1;

        p=list(map(int, p));
        tmp=img[p[1]:p[3], p[0]:p[2]];
        ret, result=cv.threshold(tmp, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU); 
        dst=cv.resize(result, dsize=winSize, interpolation=cv.INTER_CUBIC);
        descriptor=hog.compute(dst);
        testData=descriptor.reshape(-1, 1620);
        retval, res=svm.predict(testData);
        #print(int(res[0]));
        if(p[4]==int(res[0])):
            correct+=1;
        else:
            print(anno);
            print(int(res[0]), p[4]);

print("finished test");
print("result :", correct/total*100,'%');

