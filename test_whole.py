import cv2;
import numpy as np;
import os;

winSize=(30, 50);
blockSize=(10, 10);
blockStride=(5, 5);
cellSize=(5, 5);
nbins=9;

correct=0;

kernel = np.ones((3, 3), np.uint8);

str_path_dir="./data/test/strings";
img_path_dir="./data/images";

model="./front_trained/frozen_inference_graph.pb";
config="./front_trained/graph.pbtxt";

str_file_list=os.listdir(str_path_dir);
img_file_list=os.listdir(img_path_dir);

net=cv2.dnn.readNet(model, config);
if net.empty():
    print("net error");

svm=cv2.ml.SVM_load("digits_svm_model.yml");
hog=cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

def id2class(num):
    if num is 10:
        return '가';
    elif num is 11:
        return '나';
    elif num is 12:
        return '다';
    elif num is 13:
        return '라';
    elif num is 14:
        return '마';
    elif num is 15:
        return '거';
    elif num is 16:
        return '너';
    elif num is 17:
        return '더';
    elif num is 18:
        return '러';
    elif num is 19:
        return '머';
    elif num is 20:
        return '버';
    elif num is 21:
        return '서';
    elif num is 22:
        return '어';
    elif num is 23:
        return '저';
    elif num is 24:
        return '고';
    elif num is 25:
        return '노';
    elif num is 26:
        return '도';
    elif num is 27:
        return '로';
    elif num is 28:
        return '모';
    elif num is 29:
        return '보';
    elif num is 30:
        return '소';
    elif num is 31:
        return '오';
    elif num is 32:
        return '조';
    elif num is 33:
        return '구';
    elif num is 34:
        return '누';
    elif num is 35:
        return '두';
    elif num is 36:
        return '루';
    elif num is 37:
        return '무';
    elif num is 38:
        return '부';
    elif num is 39:
        return '수';
    elif num is 40:
        return '우';
    elif num is 41:
        return '주';
    elif num is 42:
        return '허';
    elif num is 43:
        return '하';
    elif num is 44:
        return '호';
    else:
        return str(num);

print("start test");
for f in img_file_list:
    print(os.path.join(img_path_dir, f));
    img=cv2.imread(img_path_dir+'/'+f, cv2.IMREAD_COLOR);
    height, width, channel=img.shape;

    blob=cv2.dnn.blobFromImage(img, 1, (300, 300), (104, 177, 123));
    net.setInput(blob);
    res=net.forward();

    detect=res.reshape(res.shape[2], res.shape[3]);
    for d in detect:
        confidence=d[2];
        if(confidence<0.5):
            continue;

        x1=int(round(d[3]*width));
        y1=int(round(d[4]*height));
        x2=int(round(d[5]*width));
        y2=int(round(d[6]*height));

        license_plate=img[y1:y2, x1:x2];
    
    if "license_plate" in locals():
        try:
            gray=cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY);
            ret, binImg=cv2.threshold(gray, 0, 255, 
                                        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU);

            '''draw, cons, hier=cv2.findContours(binImg, cv2.RETR_LIST, 
                                        cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0));
            
            print(len(con));
            for i in range(0, len(con), 1):
                license_plate=cv2.drawContours(con, i, (255, 0, 0), 2);'''

            dil=cv2.dilate(binImg, kernel, iterations=2);
            op=cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel,
                                                                iterations=3);

            cnt, labels, stats, centroids=cv2.connectedComponentsWithStats(dil);
            y_sorted=stats[np.argsort(stats[:, 1])];
            x_sorted=y_sorted[np.argsort(y_sorted[:, 0])];
            print(x_sorted);

            nums=[];
            for i, s in enumerate(x_sorted):
                cv2.rectangle(license_plate, 
                            (s[0]-3, s[1]-3, s[2]+6, s[3]+6), (0, 0, 255), 2);

                if s[2]>19 and s[2]<57 and s[3]>45 and s[3]<100 :
                    print(s);
                    cv2.rectangle(license_plate, 
                            (s[0]-3, s[1]-3, s[2]+6, s[3]+6), (0, 255, 0), 2);

                    roi=binImg[s[1]-3:s[1]-3+s[3]+6, s[0]-3:s[0]-3+s[2]+6];
                    dst=cv2.resize(roi, dsize=winSize, 
                                                interpolation=cv2.INTER_CUBIC);
                    descriptor=hog.compute(dst);
                    testData=descriptor.reshape(-1, 1620);
                    retval, num=svm.predict(testData);
                    print(int(num));
                    nums.append(id2class(int(num)));
                    print(nums);
        
            final=''.join(nums);
            str_fd=open(str_path_dir+'/'+ 
                                f.replace("jpg", "txt"), 'r', encoding='UTF8');
            str_line=str(str_fd.readline()).strip();
            print(final, str_line);

            if final == str_line:
                correct+=1;

            cv2.imshow('lp', license_plate);
            cv2.imshow('dil', dil);
            cv2.imshow('op', op);
            cv2.imshow('bin', binImg);
            cv2.waitKey(0);
            cv2.destroyAllWindows();

            del license_plate, dil, binImg;
        
        except Exception as e:
            cv2.namedWindow("resized", cv2.WINDOW_NORMAL);
            cv2.resizeWindow("resized", 960, 540);
            cv2.imshow("resized", img);
            cv2.waitKey(0);
            cv2.destroyAllWindows();

            del license_plate, binImg;

            print(str(e));

    else:
        print("can't detect license plate");

        cv2.namedWindow("resized", cv2.WINDOW_NORMAL);
        cv2.resizeWindow("resized", 960, 540);
        cv2.imshow("resized", img);
        cv2.waitKey(0);
        cv2.destroyAllWindows();

print(correct/len(img_file_list)*100, '%');
cv2.destroyAllWindows();


            
    


    