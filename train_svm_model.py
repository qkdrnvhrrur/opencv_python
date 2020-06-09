import cv2 as cv;
import os;
import numpy as np;

anno_path_dir="./data/train/annotations";
image_path_dir="./data/train/images";

anno_file_list=os.listdir(anno_path_dir);
train_image_file_list=os.listdir(image_path_dir);

#print(anno_file_list);
#print(len(anno_file_list));
#print(image_file_list);
#print(len(image_file_list));

train_hog=[];
train_label=[];

winSize=(30, 50);
blockSize=(10, 10);
blockStride=(5, 5);
cellSize=(5, 5);
nbins=9;

hog=cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

for anno in anno_file_list:
    anno_fd=open(os.path.join(anno_path_dir, anno), 'r');
    pos=[];
        
    anno_line=anno_fd.readline();
    while anno_line:
        anno_line=anno_fd.readline();
        #print(anno_line);
        if not anno_line: break;

        pos.append(anno_line.strip().split(' '));
        
    #print('pos', pos);
    img=cv.imread(os.path.join(image_path_dir, anno.replace("txt", "jpg")),
           cv.IMREAD_GRAYSCALE);

    for p in pos:
        p=list(map(int, p));
        #cv.rectangle(img, (p[0], p[1]), (p[2], p[3]), (0, 0, 0), 1);
        tmp=img[p[1]:p[3], p[0]:p[2]];
        ret, result=cv.threshold(tmp, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU); 
        #img[p[1]:p[3], p[0]:p[2]]=result;
        dst=cv.resize(result, dsize=winSize, interpolation=cv.INTER_CUBIC);
        descriptor=hog.compute(dst);
        #print(type(descriptor));
        #print(descriptor.shape);
        #print(descriptor.T.shape);
        train_hog.append(descriptor);
        train_label.append(p[4]);
        
    #cv.imshow(anno, img);
    #cv.waitKey(0);
    #cv.destroyAllWindows();
    print(anno+" finished");

#print('train hog', train_hog);
#print('train label', train_label);
trainData=np.float32(train_hog).reshape(-1, 1620);
'''print(type(trainData));
print(trainData.shape);
print(trainData);'''
trainLabels=np.array(train_label);
'''print(type(trainLabels));
print(trainLabels.shape);
print(trainLabels);'''

print("training started");

svm=cv.ml_SVM.create();
svm.setKernel(cv.ml.SVM_LINEAR);
svm.setType(cv.ml.SVM_C_SVC);
svm.setC(0.1);
svm.setGamma(1.0);
svm.train(trainData, cv.ml.ROW_SAMPLE, trainLabels);
#print("Gamma : ", svm.getGamma());
#print("C : ", svm.getC());
#svm.save("digits_svm_model.xml");
svm.save("./front_trained/digits_svm_model.yml");

print("training finished");






